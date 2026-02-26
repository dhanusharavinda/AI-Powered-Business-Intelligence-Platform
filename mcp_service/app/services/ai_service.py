from __future__ import annotations

from typing import Any, Dict, Optional

from app.services.ai_chat_service import chat


def generate_executive_report(*, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Executive AI endpoint wrapper.

    Architectural constraints:
    - AI MUST NOT query Postgres directly.
    - All intelligence MUST flow through MCP endpoints (/revenue/*, /churn/*).
    - LLM must receive ONLY structured metrics (no raw JSON, no SQL).
    """
    result = chat(
        "Provide an executive SaaS performance overview with trend analysis, risk evaluation, and recommendations.",
        session_id=session_id,
    )

    return {
        "session_id": result.get("session_id"),
        "used_endpoints": result.get("used_endpoints"),
        "metrics": result.get("structured_metrics"),
        "summary": result.get("answer"),
    }