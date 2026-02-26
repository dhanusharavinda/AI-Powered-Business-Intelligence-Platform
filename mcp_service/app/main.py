from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.routes import revenue, churn, ai

app = FastAPI(title="SaaS Intelligence MCP")

app.include_router(revenue.router)
app.include_router(churn.router)
app.include_router(ai.router)

_BASE_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(_STATIC_DIR / "index.html"))

