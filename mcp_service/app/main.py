from fastapi import FastAPI
from app.routes import revenue, churn

app = FastAPI(title="SaaS Intelligence MCP")

app.include_router(revenue.router)
app.include_router(churn.router)


@app.get("/")
def root():
    return {"message": "SaaS Intelligence MCP is running"}

