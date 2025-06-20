from fastapi import FastAPI, Query, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import mysql.connector
from mysql.connector import pooling
from budgets import (
    get_budget_data,
    generate_budget_analysis,
    get_projected_spend,
    get_department_insights,
    get_department_projected_spend,
    get_approval_trend_insights,
    get_department_approval_trend_insights
)

app = FastAPI()

# Create the connection pool at startup
pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=10,
    host='13.247.208.85',
    port='3306',
    database='vgtechde_gopaddid',
    user='vgtechde_gopaddiv2',
    password='[VZNh-]E%{6q'
)

def get_db():
    conn = pool.get_connection()
    try:
        yield conn
    finally:
        conn.close()

@app.get("/organisation_report")
def organisation_report(
    business_id: int = Query(..., description="Business ID"),
    conn = Depends(get_db)
):
    budget_data = get_budget_data(business_id, conn)
    analysis = generate_budget_analysis(budget_data)
    return JSONResponse(content=analysis)

@app.get("/budget_overview")
def budget_overview(
    business_id: int = Query(..., description="Business ID"),
    frequency: str = Query("monthly", description="Frequency: yearly, monthly, custom"),
    year: Optional[int] = Query(None, description="Year for yearly frequency"),
    month: Optional[int] = Query(None, description="Month (1-12) for monthly frequency"),
    start_date: Optional[str] = Query(None, description="Start date for custom (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for custom (YYYY-MM-DD)"),
    conn = Depends(get_db)
):
    result = get_projected_spend(
        business_id=business_id,
        conn=conn,
        frequency=frequency,
        year=year,
        month=month,
        start_date=start_date,
        end_date=end_date
    )
    currency = result.get("currency", "NGN")
    spend = result.get("projected_spend", 0.0)
    if frequency == "yearly" and year:
        period = f"{year}"
    elif frequency == "monthly" and month:
        from calendar import month_name
        period = f"{month_name[month]} {year if year else ''}".strip()
    elif frequency == "custom" and start_date and end_date:
        period = f"{start_date} to {end_date}"
    else:
        period = "the selected period"
    sentence = f"Your projected spend for this {frequency} ({period}) is {spend:,.2f} {currency}"
    return JSONResponse(content={"message": sentence})

@app.get("/department_budget_overview")
def department_budget_overview(
    business_id: int = Query(..., description="Business ID"),
    department_id: int = Query(..., description="Department ID"),
    frequency: str = Query("monthly", description="Frequency: yearly, monthly, custom"),
    year: Optional[int] = Query(None, description="Year for yearly frequency"),
    month: Optional[int] = Query(None, description="Month (1-12) for monthly frequency"),
    start_date: Optional[str] = Query(None, description="Start date for custom (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for custom (YYYY-MM-DD)"),
    conn = Depends(get_db)
):
    result = get_department_projected_spend(
        business_id=business_id,
        department_id=department_id,
        conn=conn,
        frequency=frequency,
        year=year,
        month=month,
        start_date=start_date,
        end_date=end_date
    )
    currency = result.get("currency", "NGN")
    spend = result.get("projected_spend", 0.0)
    if frequency == "yearly" and year:
        period = f"{year}"
    elif frequency == "monthly" and month:
        from calendar import month_name
        period = f"{month_name[month]} {year if year else ''}".strip()
    elif frequency == "custom" and start_date and end_date:
        period = f"{start_date} to {end_date}"
    else:
        period = "the selected period"
    sentence = f"Your department's projected spend for this {frequency} ({period}) is {spend:,.2f} {currency}"
    return JSONResponse(content={"message": sentence, "details": result})

@app.get("/department_report")
def department_report(
    business_id: int = Query(..., description="Business ID"),
    department_id: int = Query(..., description="Department ID"),
    conn = Depends(get_db)
):
    result = get_department_insights(business_id, department_id, conn)
    return JSONResponse(content=result)

@app.get("/organisation_approval_trends")
def organisation_approval_trends(
    business_id: int = Query(..., description="Business ID"),
    conn = Depends(get_db)
):
    result = get_approval_trend_insights(business_id, conn)
    return JSONResponse(content=result)

@app.get("/department_approval_trends")
def department_approval_trends(
    business_id: int = Query(..., description="Business ID"),
    department_id: int = Query(..., description="Department ID"),
    conn = Depends(get_db)
):
    result = get_department_approval_trend_insights(business_id, department_id, conn)
    return JSONResponse(content=result)