from fastapi import APIRouter

from app.db import execute_query

router = APIRouter(prefix="/churn", tags=["Churn"])


@router.get("/monthly-rate")
def get_monthly_churn_rate():
    query = "SELECT * FROM warehouse.monthly_churn_rate;"
    return execute_query(query)


@router.get("/by-plan")
def get_churn_by_plan():
    query = "SELECT * FROM warehouse.churn_by_plan;"
    return execute_query(query)


@router.get("/by-industry")
def get_churn_by_industry():
    query = "SELECT * FROM warehouse.churn_by_industry;"
    return execute_query(query)


@router.get("/avg-duration")
def get_avg_subscription_duration():
    query = "SELECT * FROM warehouse.avg_subscription_duration;"
    return execute_query(query)


@router.get("/upgrade-before-churn")
def get_upgrade_before_churn_rate():
    query = "SELECT * FROM warehouse.upgrade_before_churn_rate;"
    return execute_query(query)


@router.get("/support-correlation")
def get_support_churn_correlation():
    query = "SELECT * FROM warehouse.support_churn_correlation;"
    return execute_query(query)

