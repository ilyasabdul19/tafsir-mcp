"""Static Quran metadata: surah names and ayah -> page/juz/hizb mapping.

Source: Tanzil.info quran-data.js (CC BY 3.0) — standard Madani mushaf
boundaries (604 pages, 30 ajza', 240 hizb quarters). The tafsir DB has no
per-ayah page mapping, and the mobile app navigates by mushaf page, so the
REST facade joins this index into every verse result.
"""
from __future__ import annotations

import json
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

_META_PATH = Path(__file__).parent / "quran_meta.json"


def _absolute(surah: int, ayah: int, sura_starts: list[int]) -> int:
    """1-based absolute ayah number across the whole Quran."""
    return sura_starts[surah] + ayah


@lru_cache(maxsize=1)
def _load() -> dict:
    raw = json.loads(_META_PATH.read_text("utf-8"))

    # sura rows: [start, ayas, order, rukus, name, tname, ename, type]
    sura_starts = [0] * 115
    suras: dict[int, dict] = {}
    for i in range(1, 115):
        row = raw["sura"][i]
        sura_starts[i] = row[0]
        suras[i] = {
            "id": i,
            "arabic_name": row[4],
            "name": row[5],
            "english_name": row[6],
            "type": row[7],
            "ayas": row[1],
        }

    def boundaries(key: str) -> list[int]:
        # rows: [sura, aya] start of each unit; last row is a [115, 1] sentinel
        rows = raw[key][1:-1]
        return [_absolute(s, a, sura_starts) for s, a in (r[:2] for r in rows)]

    return {
        "suras": suras,
        "sura_starts": sura_starts,
        "page": boundaries("page"),
        "juz": boundaries("juz"),
        "hizb_quarter": boundaries("hizb_quarter"),
    }


_UTHMANI_PATH = Path(__file__).parent / "quran_uthmani.txt"


@lru_cache(maxsize=1)
def _uthmani() -> dict[tuple[int, int], str]:
    """Vocalized Uthmani text per ayah (Tanzil Project, tanzil.net).

    The tafsir DB stores only undiacritized text; this is the display copy.
    """
    verses: dict[tuple[int, int], str] = {}
    for line in _UTHMANI_PATH.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        surah, ayah, text = line.split("|", 2)
        verses[(int(surah), int(ayah))] = text
    return verses


def uthmani_text(surah: int, ayah: int) -> str | None:
    return _uthmani().get((surah, ayah))


def sura_info(surah: int) -> dict:
    return _load()["suras"][surah]


def position_of(surah: int, ayah: int) -> dict:
    """Return {page, juz, hizb} for an ayah (standard Madani mushaf)."""
    meta = _load()
    abs_no = _absolute(surah, ayah, meta["sura_starts"])
    page = bisect_right(meta["page"], abs_no) or 1
    juz = bisect_right(meta["juz"], abs_no) or 1
    quarter = bisect_right(meta["hizb_quarter"], abs_no) or 1
    return {"page": min(page, 604), "juz": min(juz, 30), "hizb": min((quarter + 3) // 4, 60)}
