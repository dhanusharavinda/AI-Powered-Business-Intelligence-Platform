from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.ai_chat_service import chat
from app.services.ai_service import generate_executive_report

router = APIRouter(prefix="/ai", tags=["AI Insights"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None


@router.post("/chat")
def chat_ai(req: ChatRequest):
    return chat(req.question, session_id=req.session_id)


@router.get("/executive-report")
def get_executive_report(session_id: Optional[str] = None):
    return generate_executive_report(session_id=session_id)