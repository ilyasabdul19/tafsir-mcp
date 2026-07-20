"""Plain REST facade over the tafsir-mcp tools, for mobile/web clients.

Registered as Starlette custom routes on the FastMCP app, so one process
serves both /mcp (LLM assistants) and /api/* (the app). Three groups:

  GET  /api/tools                  list the 13 tools + input schemas
  POST /api/tools/{name}           generic bridge: JSON body -> tool -> JSON
  GET  /api/search                 FTS verse search, app-shaped response
  GET  /api/tafsir/{surah}/{ayah}  tafsir from one or more sources
  GET  /api/ayah/{surah}/{ayah}    ayah text (+ optional tajweed/irab)
  POST /api/ask                    AI search: Claude + all tools, app-shaped

/api/ask requires OPENROUTER_API_KEY (returns 503 without it); model and
gateway are configurable via ASK_MODEL / ASK_BASE_URL (any OpenAI-compatible
endpoint works). If APP_API_KEY is set, /api/ask also requires a matching
X-App-Key header.

Attribution requirement (data license CC BY 4.0): responses include
`attribution` naming Tafsir Center for Quranic Studies.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from tafsir.tools.ayah import get_ayah, get_ayah_nuzool, get_ayah_tafsir
from tafsir.tools.qeraat import compare_qeraat
from tafsir.tools.search import search_quran_text, search_tafsir
from tafsir.tools.stats import (
    get_page_fawaed,
    get_quran_statistics,
    get_surah_statistics_summary,
)
from tafsir.tools.surah import get_surah_info
from tafsir.tools.word import get_root_statistics, get_word_analysis, search_by_root

from quran_meta import position_of, sura_info, uthmani_text

ATTRIBUTION = (
    "Tafsir Center for Quranic Studies — tafsir.net (data: CC BY 4.0); "
    "Quran text: Tanzil Project — tanzil.net"
)

PAGE_SIZE = 20
FTS_POOL = 100  # max candidates fetched per query; paginated client-side

# name -> callable, mirrors the MCP tool names exactly
TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "fetch_ayah": get_ayah,
    "fetch_tafsir": get_ayah_tafsir,
    "fetch_nuzool_reason": get_ayah_nuzool,
    "fetch_surah_info": get_surah_info,
    "get_surah_statistics": get_surah_statistics_summary,
    "analyze_word": get_word_analysis,
    "find_root_occurrences": search_by_root,
    "get_root_stats": get_root_statistics,
    "get_qeraat_variants": compare_qeraat,
    "search_quran_text": search_quran_text,
    "search_in_tafsir": search_tafsir,
    "get_quran_overview": get_quran_statistics,
    "get_page_fawaed": get_page_fawaed,
}


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"success": False, "error": message}, status_code=status)


# ── Access control & rate limiting ───────────────────────────────────────────
#
# If APP_API_KEY is set, every /api/* route requires a matching X-App-Key
# header. Rate limits are sliding-window, in-memory, per client IP (per
# machine — good enough behind Fly's load balancer for this traffic level).

_RATE_LIMITS: dict[str, tuple[int, int]] = {
    "api": (120, 60),  # general endpoints: 120 req / minute
    "ask": (10, 60),  # LLM endpoint: 10 req / minute
    "ask_day": (200, 86400),  # LLM endpoint: 200 req / day
}
_BUCKETS: dict[tuple[str, str], deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    return (
        request.headers.get("fly-client-ip")
        or (request.client.host if request.client else "unknown")
    )


def _rate_limited(request: Request, *buckets: str) -> JSONResponse | None:
    ip = _client_ip(request)
    now = time.monotonic()
    for bucket in buckets:
        max_requests, window = _RATE_LIMITS[bucket]
        entries = _BUCKETS[(bucket, ip)]
        while entries and now - entries[0] > window:
            entries.popleft()
        if len(entries) >= max_requests:
            retry_after = int(window - (now - entries[0])) + 1
            response = _error(429, "rate limit exceeded, slow down")
            response.headers["Retry-After"] = str(retry_after)
            return response
    for bucket in buckets:
        _BUCKETS[(bucket, ip)].append(now)
    return None


def _guard(request: Request, *buckets: str) -> JSONResponse | None:
    """App-key check + rate limit; returns an error response or None."""
    app_key = os.getenv("APP_API_KEY", "").strip()
    if app_key and request.headers.get("x-app-key") != app_key:
        return _error(401, "invalid or missing X-App-Key")
    return _rate_limited(request, *buckets)


def _verse_result(surah: int, ayah: int, text: str) -> dict:
    """Shape one verse the way the mobile app's search contract expects.

    Display text prefers the vocalized Tanzil Uthmani copy; the tafsir DB's
    undiacritized text is only a fallback.
    """
    sura = sura_info(surah)
    return {
        "aya": {"id": ayah, "text": uthmani_text(surah, ayah) or text},
        "identifier": {
            "sura_id": surah,
            "aya_id": ayah,
            "sura_name": sura["name"],
            "sura_arabic_name": sura["arabic_name"],
        },
        "position": position_of(surah, ayah),
        "sura": {
            "id": surah,
            "name": sura["name"],
            "arabic_name": sura["arabic_name"],
            "english_name": sura["english_name"],
            "type": sura["type"],
            "ayas": sura["ayas"],
        },
    }


# Clitic prefixes that attach to Quranic words: definite article, conjunctions,
# prepositions, and their combinations. FTS5 tokenizes clitics as part of the
# word ("بالصبر"), so a bare query like "الصبر" finds nothing without expansion.
_CLITICS = ("", "ال", "و", "ف", "ب", "ل", "ك", "وال", "فال", "بال", "كال", "لل", "ولل", "وب", "ول")


def _expand_fts_query(query: str) -> str:
    """Turn plain words into an FTS5 expression tolerant of attached clitics.

    "الصبر" -> "(صبر* OR الصبر* OR وصبر* OR بالصبر* OR ...)"; multi-word input
    becomes AND-ed groups. FTS operators in the input are kept as-is so the
    /api/ask model can pass through hand-crafted expressions.
    """
    from tafsir.normalize import normalize_arabic

    if any(op in query for op in ('"', "*", " OR ", " AND ", " NOT ")):
        return query
    words = [w for w in normalize_arabic(query).split() if w]
    groups = []
    for word in words:
        stem = word[2:] if word.startswith("ال") and len(word) > 4 else word
        variants = dict.fromkeys([word] + [c + stem for c in _CLITICS])
        groups.append("(" + " OR ".join(f"{v}*" for v in variants) + ")")
    return " ".join(groups) or query


def _paginated_search(query: str, page: int) -> dict:
    """Run FTS once, slice into app-style pages of PAGE_SIZE."""
    hits = search_quran_text(query=_expand_fts_query(query), limit=FTS_POOL)
    total = len(hits)
    nb_pages = max(1, math.ceil(total / PAGE_SIZE))
    page = max(1, min(page, nb_pages))
    window = hits[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
    ayas = {
        str(i): _verse_result(h["surah"], h["ayah"], h["text"])
        for i, h in enumerate(window)
    }
    return {
        "ayas": ayas,
        "interval": {"page": page, "nb_pages": nb_pages, "total": total},
    }


# ── LLM-powered /api/ask (via OpenRouter, OpenAI-compatible) ─────────────────

ASK_MODEL = os.getenv("ASK_MODEL", "anthropic/claude-haiku-4.5")
ASK_BASE_URL = os.getenv("ASK_BASE_URL", "https://openrouter.ai/api/v1")
ASK_MAX_TURNS = 8

SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": (
        "Submit the final answer to the user's question. ALWAYS finish by "
        "calling this tool exactly once, after consulting the Quran tools."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "explain": {
                "type": "string",
                "description": (
                    "Concise answer to the user's question, in the user's "
                    "language, grounded ONLY in tool results. 2-6 sentences."
                ),
            },
            "generated_query": {
                "type": "string",
                "description": (
                    "ONE short Arabic keyword or phrase (1-3 words max) that "
                    "literally appears in relevant verses and returned good "
                    "results from search_quran_text. Words are AND-ed: more "
                    "words = fewer results, so prefer a single strong word."
                ),
            },
            "query_language": {"type": "string", "description": "BCP-47 of the user's query, e.g. ar, en"},
            "verses": {
                "type": "array",
                "description": "Up to 10 specific verses to pin at the top of results, most relevant first.",
                "items": {
                    "type": "object",
                    "properties": {
                        "surah": {"type": "integer", "minimum": 1, "maximum": 114},
                        "ayah": {"type": "integer", "minimum": 1},
                    },
                    "required": ["surah", "ayah"],
                },
            },
        },
        "required": ["explain", "generated_query", "query_language"],
    },
}

ASK_SYSTEM = f"""You answer questions about the Quran for a mobile app, using ONLY the provided tools as the source of truth (verified data from Tafsir Center for Quranic Studies). Never answer religious content from memory — always consult the tools first.

STRICT GROUNDING: your training knowledge of the Quran is off-limits. Every verse text, every [surah:ayah] number, and every tafsir attribution in your answer (including the verses you pin in submit_answer) must come verbatim from a tool result in this conversation — verify with search_quran_text or fetch_ayah before citing. If the tools return nothing relevant, say so honestly in explain instead of answering from memory.

Workflow:
1. Understand the question (Arabic or English or other).
2. Call search_quran_text with a diacritic-free Arabic phrase likely to appear in relevant verses. Refine and retry if results are poor. Use fetch_tafsir / analyze_word / fetch_nuzool_reason / find_root_occurrences when the question is about meaning, a word, or revelation context.
3. Finish by calling submit_answer exactly once: a short grounded answer (same language as the question), the best Arabic FTS phrase as generated_query, and pinned verses if specific ayahs directly answer the question.

Rules: keep explain concise and neutral in tone; cite tafsir sources by name when you rely on them (attribution: {ATTRIBUTION}); if nothing relevant is found, say so honestly in explain and still provide your best generated_query."""


def _openai_tools() -> list[dict]:
    """Build OpenAI-style tool definitions from the registry's own docstrings."""
    import inspect

    tools = []
    for name, fn in TOOL_REGISTRY.items():
        schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for pname, param in inspect.signature(fn).parameters.items():
            ann = param.annotation
            ptype: dict[str, Any]
            if ann is int or "int" in str(ann):
                ptype = {"type": "integer"}
            elif "list" in str(ann):
                items = {"type": "integer"} if "int" in str(ann) else {"type": "string"}
                ptype = {"type": "array", "items": items}
            else:
                ptype = {"type": "string"}
            schema["properties"][pname] = ptype
            if param.default is inspect.Parameter.empty:
                schema["required"].append(pname)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (fn.__doc__ or name).strip()[:1000],
                    "parameters": schema,
                },
            }
        )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": SUBMIT_TOOL["name"],
                "description": SUBMIT_TOOL["description"],
                "parameters": SUBMIT_TOOL["input_schema"],
            },
        }
    )
    return tools


async def _run_ask(message: str) -> dict:
    """LLM tool-use loop via OpenRouter; returns the submit_answer payload."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=ASK_BASE_URL,
        # .strip(): secrets pasted via `fly secrets set` can carry a trailing
        # newline, which silently corrupts the Authorization header.
        api_key=os.environ["OPENROUTER_API_KEY"].strip(),
    )
    tools = _openai_tools()
    messages: list[dict] = [
        {"role": "system", "content": ASK_SYSTEM},
        {"role": "user", "content": message},
    ]

    for _ in range(ASK_MAX_TURNS):
        response = await client.chat.completions.create(
            model=ASK_MODEL,
            max_tokens=2000,
            tools=tools,
            messages=messages,
        )
        choice = response.choices[0].message
        tool_calls = choice.tool_calls or []
        if not tool_calls:
            # Model answered in prose without submitting — nudge it once.
            messages.append({"role": "assistant", "content": choice.content or ""})
            messages.append(
                {"role": "user", "content": "Call submit_answer now with your final answer."}
            )
            continue

        messages.append(
            {
                "role": "assistant",
                "content": choice.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            }
        )

        submitted: dict | None = None
        for call in tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if call.function.name == SUBMIT_TOOL["name"]:
                submitted = args
                content = "ok"
            else:
                fn = TOOL_REGISTRY.get(call.function.name)
                try:
                    out = await asyncio.to_thread(fn, **args) if fn else {"error": "unknown tool"}
                except Exception as exc:  # tool input errors come back to the model
                    out = {"error": str(exc)}
                content = json.dumps(out, ensure_ascii=False, default=str)[:20000]
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )
        if submitted is not None:
            return submitted

    # Round budget exhausted without submit_answer (seen with some models on
    # broad thematic questions): salvage the last prose + searched query
    # instead of failing the request.
    last_text = next(
        (
            m["content"]
            for m in reversed(messages)
            if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"]
        ),
        "",
    )
    last_query = ""
    for m in reversed(messages):
        for call in m.get("tool_calls") or []:
            if call["function"]["name"] == "search_quran_text":
                try:
                    last_query = json.loads(call["function"]["arguments"]).get("query", "")
                except json.JSONDecodeError:
                    pass
                break
        if last_query:
            break
    if last_text or last_query:
        return {"explain": last_text, "generated_query": last_query, "query_language": ""}
    raise RuntimeError("model did not produce a final answer")


def _ask_response(submitted: dict, page: int = 1) -> dict:
    search = _paginated_search(submitted.get("generated_query", ""), page)

    # Pin the model's explicitly chosen verses at the top of page 1.
    pinned = submitted.get("verses") or []
    if page == 1 and pinned:
        seen = {
            (v["identifier"]["sura_id"], v["identifier"]["aya_id"])
            for v in search["ayas"].values()
        }
        ordered: list[dict] = []
        for ref in pinned[:10]:
            key = (ref["surah"], ref["ayah"])
            if key in seen:
                seen.discard(key)  # already present: will be re-added in pinned order
            text = uthmani_text(ref["surah"], ref["ayah"])
            if text is None:  # invalid ref hallucinated by the model
                continue
            ordered.append(_verse_result(ref["surah"], ref["ayah"], text))
        for verse in search["ayas"].values():
            key = (verse["identifier"]["sura_id"], verse["identifier"]["aya_id"])
            if key in seen:
                ordered.append(verse)
        search["ayas"] = {str(i): v for i, v in enumerate(ordered[:PAGE_SIZE])}
        # FTS may have matched nothing even when verses are pinned; keep the
        # interval consistent with what is actually being returned.
        search["interval"]["total"] = max(search["interval"]["total"], len(ordered))

    return {
        "success": True,
        "attribution": ATTRIBUTION,
        "ai": {
            "explain": submitted.get("explain", ""),
            "generated_query": submitted.get("generated_query", ""),
            "sort_by": "relevance",
            "proofread_user_query": "",
            "query_language": submitted.get("query_language", ""),
        },
        "search": search,
    }


# ── Streaming /api/chat (ChatGPT-style conversational interface) ─────────────

CHAT_MAX_MESSAGES = 30
CHAT_MAX_CHARS = 4000
CHAT_MAX_ROUNDS = 8

CHAT_SYSTEM = f"""You are a warm, knowledgeable Quran study companion inside a mobile app, in an ongoing conversation. Answer in the user's language (Arabic or English).

STRICT GROUNDING — this is the most important rule and it has no exceptions:
- The tools are your ONLY source for Quranic content: verse texts, verse numbers, tafsir, word meanings, qira'at, revelation context, statistics. Your own training knowledge of the Quran is OFF-LIMITS — treat it as unreliable and never use it, even for verses you are certain about, even for al-Fatiha.
- Never write a verse's text unless that exact text came back from a tool in THIS conversation. Copy it character-for-character from the tool result — no completing, trimming, or "fixing" from memory.
- Never write a [surah:ayah] reference unless a tool returned that exact surah and ayah number. If you recall a verse but haven't verified it, call search_quran_text or fetch_ayah first.
- Never attribute a statement to a tafsir scholar unless it came from fetch_tafsir/search_in_tafsir output in this conversation.
- If the tools return nothing relevant, say plainly that you could not find it in the verified database — do NOT fill the gap from memory. An honest "لم أجد" is always better than an unverified answer.

Style:
- Conversational and concise, like a thoughtful teacher. Prefer short answers; expand only when asked.
- When you cite a specific verse, include an inline reference in the exact form [surah:ayah], e.g. [2:155] — the app turns these into tappable links. Quote the ayah text itself when it is central to the answer.
- Name tafsir sources when you rely on them (e.g. "قال السعدي..."). Attribution: {ATTRIBUTION}.
- Use plain text with occasional **bold**; no headers, no lists unless the user asks for an enumeration.
- Never dump very long tafsir texts; summarize and offer the full text on request.
- Politely decline questions unrelated to the Quran, Islam, or this app."""


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _chat_stream(history: list[dict]):
    """Yield SSE events: delta (text), status (tool activity), done / error."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=ASK_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"].strip(),
    )
    tools = _openai_tools()[:-1]  # all 13 registry tools, minus submit_answer
    messages: list[dict] = [{"role": "system", "content": CHAT_SYSTEM}, *history]

    try:
        for _ in range(CHAT_MAX_ROUNDS):
            stream = await client.chat.completions.create(
                model=ASK_MODEL,
                max_tokens=2000,
                tools=tools,
                messages=messages,
                stream=True,
            )
            content_parts: list[str] = []
            calls: dict[int, dict] = {}
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    content_parts.append(delta.content)
                    yield _sse({"type": "delta", "text": delta.content})
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

            if not calls:
                yield _sse({"type": "done"})
                return

            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(content_parts) or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in calls.values()
                    ],
                }
            )
            for c in calls.values():
                yield _sse({"type": "status", "tool": c["name"]})
                fn = TOOL_REGISTRY.get(c["name"])
                try:
                    args = json.loads(c["arguments"] or "{}")
                    out = await asyncio.to_thread(fn, **args) if fn else {"error": "unknown tool"}
                except Exception as exc:
                    out = {"error": str(exc)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": c["id"],
                        "content": json.dumps(out, ensure_ascii=False, default=str)[:20000],
                    }
                )
        yield _sse({"type": "done"})  # round budget exhausted; text so far stands
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})


# ── Route registration ───────────────────────────────────────────────────────


def register_routes(mcp) -> None:  # noqa: C901
    @mcp.custom_route("/api/tools", methods=["GET"])
    async def list_tools(request: Request) -> JSONResponse:
        if (denied := _guard(request, "api")) is not None:
            return denied
        return JSONResponse(
            {
                "success": True,
                "attribution": ATTRIBUTION,
                "tools": [
                    {"name": name, "description": (fn.__doc__ or "").strip()}
                    for name, fn in TOOL_REGISTRY.items()
                ],
            }
        )

    @mcp.custom_route("/api/tools/{name}", methods=["POST"])
    async def call_tool(request: Request) -> JSONResponse:
        if (denied := _guard(request, "api")) is not None:
            return denied
        name = request.path_params["name"]
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            return _error(404, f"unknown tool: {name}")
        try:
            args = await request.json() if await request.body() else {}
        except json.JSONDecodeError:
            return _error(400, "invalid JSON body")
        try:
            result = await asyncio.to_thread(fn, **args)
        except TypeError as exc:
            return _error(400, str(exc))
        except Exception as exc:
            return _error(500, str(exc))
        return JSONResponse(
            {"success": True, "attribution": ATTRIBUTION, "result": result}
        )

    @mcp.custom_route("/api/search", methods=["GET"])
    async def search(request: Request) -> JSONResponse:
        if (denied := _guard(request, "api")) is not None:
            return denied
        query = request.query_params.get("q", "").strip()
        if not query:
            return _error(400, "missing query param: q")
        try:
            page = max(1, int(request.query_params.get("page", "1")))
        except ValueError:
            return _error(400, "page must be an integer")
        result = await asyncio.to_thread(_paginated_search, query, page)
        return JSONResponse(
            {"success": True, "attribution": ATTRIBUTION, "search": result}
        )

    @mcp.custom_route("/api/ayah/{surah:int}/{ayah:int}", methods=["GET"])
    async def ayah(request: Request) -> JSONResponse:
        if (denied := _guard(request, "api")) is not None:
            return denied
        include = [
            part
            for part in request.query_params.get("include", "").split(",")
            if part in ("tajweed", "irab")
        ]
        try:
            result = await asyncio.to_thread(
                get_ayah, request.path_params["surah"], request.path_params["ayah"], include
            )
        except Exception as exc:
            return _error(400, str(exc))
        result["position"] = position_of(result["surah"], result["ayah"])
        return JSONResponse(
            {"success": True, "attribution": ATTRIBUTION, "result": result}
        )

    @mcp.custom_route("/api/tafsir/{surah:int}/{ayah:int}", methods=["GET"])
    async def tafsir(request: Request) -> JSONResponse:
        if (denied := _guard(request, "api")) is not None:
            return denied
        sources = [
            s for s in request.query_params.get("sources", "saadi").split(",") if s
        ]
        try:
            result = await asyncio.to_thread(
                get_ayah_tafsir, request.path_params["surah"], request.path_params["ayah"], sources
            )
        except Exception as exc:
            return _error(400, str(exc))
        return JSONResponse(
            {"success": True, "attribution": ATTRIBUTION, "result": result}
        )

    @mcp.custom_route("/api/chat", methods=["POST"])
    async def chat(request: Request) -> JSONResponse | StreamingResponse:
        if (denied := _guard(request, "api")) is not None:
            return denied
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error(400, "invalid JSON body")
        raw = body.get("messages")
        if not isinstance(raw, list) or not raw:
            return _error(400, "missing field: messages")
        history = []
        for m in raw[-CHAT_MAX_MESSAGES:]:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role not in ("user", "assistant") or not content:
                return _error(400, "messages must be {role: user|assistant, content}")
            history.append({"role": role, "content": content[:CHAT_MAX_CHARS]})
        if history[-1]["role"] != "user":
            return _error(400, "last message must be from the user")
        if not os.getenv("OPENROUTER_API_KEY"):
            return _error(503, "AI chat unavailable: OPENROUTER_API_KEY not configured")
        if (denied := _rate_limited(request, "ask", "ask_day")) is not None:
            return denied
        return StreamingResponse(
            _chat_stream(history),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @mcp.custom_route("/api/ask", methods=["POST"])
    async def ask(request: Request) -> JSONResponse:
        if (denied := _guard(request, "api")) is not None:
            return denied

        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error(400, "invalid JSON body")

        generated_query = (body.get("generated_query") or "").strip()
        page = max(1, int(body.get("page") or 1))

        # Pagination path: no LLM, just re-run FTS on the stored query.
        if generated_query:
            search = await asyncio.to_thread(_paginated_search, generated_query, page)
            return JSONResponse(
                {
                    "success": True,
                    "attribution": ATTRIBUTION,
                    "ai": {
                        "explain": "",
                        "generated_query": generated_query,
                        "sort_by": "relevance",
                        "proofread_user_query": "",
                        "query_language": "",
                    },
                    "search": search,
                }
            )

        message = (body.get("message") or "").strip()
        if not message:
            return _error(400, "missing field: message (or generated_query)")
        if len(message) > 500:
            return _error(400, "message too long (max 500 chars)")
        if not os.getenv("OPENROUTER_API_KEY"):
            return _error(503, "AI search unavailable: OPENROUTER_API_KEY not configured")
        if (denied := _rate_limited(request, "ask", "ask_day")) is not None:
            return denied

        try:
            submitted = await _run_ask(message)
        except Exception as exc:
            return _error(502, f"AI search failed: {exc}")
        response = await asyncio.to_thread(_ask_response, submitted, 1)
        return JSONResponse(response)
