"""HTTP entry point serving both MCP (/mcp) and the REST facade (/api/*).

Extends deploy/server_http.py's setup (see its docstring for the DNS
rebinding rationale) with the plain REST routes mobile clients use.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

from tafsir.server import mcp

from rest_api import register_routes


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "tafsir-mcp"})


register_routes(mcp)


def main() -> None:
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(os.getenv("PORT", 7860))
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
