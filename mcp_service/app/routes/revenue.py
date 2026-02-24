from fastapi import APIRouter

from app.db import execute_query

router = APIRouter(prefix="/revenue", tags=["Revenue"])


@router.get("/monthly")
def get_monthly_mrr():
    query = "SELECT * FROM warehouse.monthly_mrr;"
    return execute_query(query)


@router.get("/by-plan")
def get_revenue_by_plan():
    query = "SELECT * FROM warehouse.revenue_by_plan;"
    return execute_query(query)


@router.get("/by-region")
def get_revenue_by_region():
    query = "SELECT * FROM warehouse.revenue_by_region;"
    return execute_query(query)


@router.get("/churned")
def get_churned_mrr():
    query = "SELECT * FROM warehouse.churned_mrr;"
    return execute_query(query)

