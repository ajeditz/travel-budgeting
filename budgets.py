import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
import json
import openai
import os
from pprint import pprint
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()


def get_budget_data(business_id, budget_type=None):
    """
    Fetch budget data for a specific business and (optionally) budget type.
    
    Args:
        business_id (int): The ID of the business
        budget_type (str): The type of budget (default: 'organization')
        
    Returns:
        dict: Budget data with metrics for analysis
    """
    try:
        # Database connection
        conn = mysql.connector.connect(
            host='13.247.208.85',
            port='3306',
            database='vgtechde_gopaddid',
            user='vgtechde_gopaddiv2',
            password='[VZNh-]E%{6q'
        )
        
        cursor = conn.cursor(dictionary=True)
        
        # Build query dynamically based on budget_type
        query = """
        SELECT 
            b.name, 
            b.amount, 
            b.amount_spent, 
            b.start_date, 
            b.end_date, 
            b.period,
            b.status,
            b.currency,
            b.department_id,
            d.name as department_name,
            b.type as budget_type
        FROM 
            budgets b
        LEFT JOIN
            departments d ON b.department_id = d.id AND d.business_id = b.business_id
        WHERE 
            b.business_id = %s 
            AND b.active = 1
        """
        params = [business_id]
        if budget_type:
            query += " AND b.type = %s"
            params.append(budget_type)

        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame(results)

        # Convert Decimal columns to float for JSON serialization
        if not df.empty:
            for col in ['amount', 'amount_spent']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
        
        # Convert date strings to datetime objects
        if not df.empty:
            df['start_date'] = pd.to_datetime(df['start_date'])
            df['end_date'] = pd.to_datetime(df['end_date'])
            
            # Only include budgets with end_date in the last N years
            years_to_keep = 3
            cutoff_date = datetime.now() - timedelta(days=365 * years_to_keep)
            df = df[df['end_date'] >= cutoff_date]
            
            # Calculate additional metrics
            df['remaining_budget'] = df['amount'] - df['amount_spent']
            df['utilization_percentage'] = (df['amount_spent'] / df['amount'] * 100).round(2)
            
            # Calculate days elapsed and remaining
            current_date = datetime.now()
            df['days_elapsed'] = (current_date - df['start_date']).dt.days
            df['days_remaining'] = (df['end_date'] - current_date).dt.days
            df['days_total'] = (df['end_date'] - df['start_date']).dt.days
            
            # Calculate burn rate and projection
            df['daily_burn_rate'] = df.apply(
                lambda x: x['amount_spent'] / max(x['days_elapsed'], 1), axis=1
            )
            
            df['projected_total_spend'] = df.apply(
                lambda x: x['amount_spent'] + (x['daily_burn_rate'] * x['days_remaining']), 
                axis=1
            )
            
            df['projected_over_under'] = df['amount'] - df['projected_total_spend']
            
            # Format dates for output
            df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')
            df['end_date'] = df['end_date'].dt.strftime('%Y-%m-%d')

            # Only keep top 20 budgets by amount_spent
            df = df.sort_values('amount_spent', ascending=False).head(20)

            # Summarize small budgets
            top_budgets = df.sort_values('amount_spent', ascending=False).head(10)
            other_budgets = df.iloc[10:]
            summary_row = {
                'name': 'Other Departments',
                'amount': other_budgets['amount'].sum(),
                'amount_spent': other_budgets['amount_spent'].sum(),
                # ...other fields as needed...
            }
            final_df = pd.concat([top_budgets, pd.DataFrame([summary_row])], ignore_index=True)

            # Calculate department projections for next month, quarter, year
            projections = {}
            periods = {
                'month': 30,
                'quarter': 90,
                'year': 365
            }
            for period, days in periods.items():
                final_df[f'projected_spend_{period}'] = final_df['amount_spent'] + (final_df['daily_burn_rate'] * days)
                dept_proj = final_df.groupby(['department_id', 'department_name'])[f'projected_spend_{period}'].sum()
                if not dept_proj.empty:
                    top_dept = dept_proj.idxmax()
                    top_value = dept_proj.max()
                    projections[period] = {
                        'department_id': top_dept[0],
                        'department_name': top_dept[1],
                        'projected_spend': float(top_value)
                    }
                else:
                    projections[period] = None
            
            # Prepare summary statistics
            summary = {
                'business_id': business_id,
                'budget_type': budget_type,
                'total_budgets': len(final_df),
                'total_allocated_amount': final_df['amount'].sum(),
                'total_spent_amount': final_df['amount_spent'].sum(),
                'overall_utilization_percentage': round(float(final_df['amount_spent'].sum()) / float(final_df['amount'].sum()) * 100, 2),
                'currency': final_df['currency'].iloc[0] if 'currency' in final_df.columns and not final_df.empty else None,
                'budgets': final_df.to_dict('records'),
                'department_projections': projections
            }
            
            return summary
        else:
            return {
                'business_id': business_id,
                'budget_type': budget_type,
                'total_budgets': 0,
                'message': 'No active budgets found for this business and type.'
            }
            
    except mysql.connector.Error as err:
        return {
            'error': f"Database error: {err}"
        }
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

# @lru_cache(maxsize=128)
def generate_budget_analysis(budget_data):
    """
    Generate budget analysis and projections using an LLM.
    
    Args:
        budget_data (dict): The processed budget data
        
    Returns:
        dict: Analysis and projections
    """
    # Skip if there's an error or no budgets
    if 'error' in budget_data or budget_data.get('total_budgets', 0) == 0:
        return {"error": "No valid budget data to analyze"}
    
    pprint(budget_data)
    
    # Calculate current month, quarter and year
    current_date = datetime.now()
    next_month = (current_date + timedelta(days=30)).strftime('%B %Y')
    next_quarter_end = current_date + timedelta(days=90)
    next_quarter = f"Q{(next_quarter_end.month-1)//3+1} {next_quarter_end.year}"
    next_year = (current_date + timedelta(days=365)).strftime('%Y')
    
    # Extract the currency from the data
    currency = budget_data.get('currency', 'NGN')
    dept_proj = budget_data.get('department_projections', {})

    # Calculate company-level projections directly from budget_data['budgets']
    df = pd.DataFrame(budget_data['budgets'])
    projections = {}
    if not df.empty:
        for period in ['month', 'quarter', 'year']:
            col = f'projected_spend_{period}'
            projections[period] = df[col].sum() if col in df.columns else 0.0

    # Compose department projection summary (as before)
    dept_summary = ""
    for period, proj in dept_proj.items():
        if proj:
            dept_summary += (
                f"\n- {proj['department_name']} is projected to remain the top spending department, "
                f"with an estimated {currency} {proj['projected_spend']:,.2f} for the next {period}."
            )
    
    # Format the prompt for the LLM
    prompt = f"""
    You are an AI financial analyst specializing in travel agency budget analysis. 
    Analyze this budget data for a travel agency and provide clear, specific spending projections for the next month, quarter, and year.
    
    Company Data:
    - Business ID: {budget_data['business_id']}
    - Total allocated budget: {budget_data['total_allocated_amount']:,.2f} {currency}
    - Total spent to date: {budget_data['total_spent_amount']:,.2f} {currency}
    - Overall utilization: {budget_data['overall_utilization_percentage']}%
    {dept_summary}
    
    Individual Budgets:
    """
    
    # Add each budget to the prompt
    for budget in budget_data.get('budgets', []):
        prompt += f"""
        * {budget['name']}:
          - Allocated: {budget['amount']:,.2f} {currency}
          - Spent: {budget['amount_spent']:,.2f} {currency}
          - Utilization: {budget['utilization_percentage']}%
          - Period: {budget['period']}
          - Timeline: {budget['start_date']} to {budget['end_date']}
          - Daily burn rate: {budget['daily_burn_rate']:,.2f} {currency}/day
        """

    # Add specific instructions to get the desired format
    prompt += f"""
    Based on this data, provide:
    
    1. A projection for spending BY THE END OF {next_month} (next month) in this exact format:
       "At the current pace, the company is projected to spend X {currency} by the end of {next_month}."
    
    2. A projection for spending BY THE END OF {next_quarter} (next quarter) in this exact format:
       "At the current pace, the company is projected to spend X {currency} by the end of {next_quarter}."
    
    3. A projection for spending BY THE END OF {next_year} (next year) in this exact format:
       "At the current pace, the company is projected to spend X {currency} by the end of {next_year}."
    
    4. 2-3 key insights about the overall budget utilization and spending patterns.
    
    5. 1-2 actionable recommendations for budget management.
    
    Only include the direct responses in the specified formats. Do not include any additional explanation or analysis before the projections.
    """
    
    prompt += f"""
    Please provide your output in the following JSON format:

    {{
        "organisation_insights": [
            "Insight 1 (with a number/figure)",
            "Insight 2 (with a number/figure)",
            "Insight 3 (with a number/figure)",
            "Insight 4 (with a number/figure)"
        ],
        "highest_spending_department_insights": [
            "Insight 1 (with a number/figure, department name, and projections)",
            "Insight 2 (with a number/figure, department name, and projections)",
            "Insight 3 (with a number/figure, department name, and projections)",
            "Insight 4 (with a number/figure, department name, and projections)"
        ]
    }}

    Organisation insights should be specific and helpful to the organisation as a whole. 
    Highest spending department insights should focus on the department with the highest projected spend, including predictions for the next month, quarter, and year.
    """
    
    prompt += f"""
You must reply ONLY with a valid JSON object in the following format, and nothing else:

{{
    "organisation_insights": [
        "A single, concise insight with a number/figure (e.g., overall utilization, burn rate, trend, outlier, or projection).",
        "A different type of insight (e.g., trend, anomaly, or comparison).",
        "Another unique insight (e.g., cost-saving opportunity, efficiency, or risk).",
        "A projection for the next month.",
        "A projection for the next quarter.",
        "A projection for the next year."
    ],
    "highest_spending_department_insights": [
        "A single, concise insight about the highest spending department, with a number/figure and department name (e.g., burn rate, utilization, or trend).",
        "A projection for the next month for this department.",
        "A projection for the next quarter for this department.",
        "A projection for the next year for this department.",
        "A unique insight about this department (e.g., how it compares to others, or a risk/opportunity)."
    ]
}}

**Instructions:**
- Each insight must be a single, atomic point (do not combine multiple insights in one item).
- Do not include any text, explanation, or formatting outside the JSON object.
- Each insight must include a number or figure.
- Ensure insights are of different types (not just projections).
- For department insights, always mention the department name and a projection (for month, quarter, or year) where relevant.
"""
    
    # Call the LLM API (using OpenAI as an example)
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Make sure your OPENAI_API_KEY is set in your environment
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial analyst specializing in travel agency budgets."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        analysis = response.choices[0].message.content
        
        # Split the analysis into sections for easier consumption by your frontend
        analysis_parts = analysis.split("\n\n")
        
        # Use your own projections instead of extract_projection:
        result = {
            "projection_month": f"At the current pace, the company is projected to spend {projections['month']:,.2f} {currency} by the end of {next_month}.",
            "projection_quarter": f"At the current pace, the company is projected to spend {projections['quarter']:,.2f} {currency} by the end of {next_quarter}.",
            "projection_year": f"At the current pace, the company is projected to spend {projections['year']:,.2f} {currency} by the end of {next_year}.",
            "insights": [part.strip() for part in analysis_parts if "insight" in part.lower() or "key" in part.lower()],
            "recommendations": [part.strip() for part in analysis_parts if "recommendation" in part.lower() or "action" in part.lower()],
            "full_analysis": analysis
        }
        
        import json

        import re
        import json

        def extract_json_from_llm_output(output):
            match = re.search(r'\{[\s\S]*\}', output)
            if match:
                return match.group(0)
            return None

        # ...after getting analysis from the LLM...
        json_str = extract_json_from_llm_output(analysis)
        if json_str:
            try:
                insights_json = json.loads(json_str)
                result = {
                    "organisation_insights": insights_json.get("organisation_insights", []),
                    "highest_spending_department_insights": insights_json.get("highest_spending_department_insights", [])
                }
            except Exception as e:
                result = {
                    "organisation_insights": [],
                    "highest_spending_department_insights": [],
                    "raw_llm_output": analysis,
                    "error": f"Could not parse LLM output as JSON: {e}"
                }
        else:
            result = {
                "organisation_insights": [],
                "highest_spending_department_insights": [],
                "raw_llm_output": analysis,
                "error": "No JSON object found in LLM output."
            }
        return result
    
    except Exception as e:
        # For testing purposes, return mock projections if OpenAI API is not configured
        return {
            "projection_month": f"At the current pace, the company is projected to spend 0.00 {currency} by the end of {next_month}.",
            "projection_quarter": f"At the current pace, the company is projected to spend 0.00 {currency} by the end of {next_quarter}.",
            "projection_year": f"At the current pace, the company is projected to spend 0.00 {currency} by the end of {next_year}.",
            "insights": [
                "All budgets are currently showing 0% utilization, indicating no spending activity has been recorded.",
                "Multiple budget periods overlap, which may cause confusion in financial reporting."
            ],
            "recommendations": [
                "Establish regular budget tracking to ensure spend data is being properly recorded in the system.",
                "Review and consolidate overlapping budget periods to improve financial planning."
            ],
            "error": str(e)
        }

def extract_projection(analysis, period_type):
    """
    Extract the projection for a specific period from the analysis text.
    """
    lines = analysis.split("\n")
    for line in lines:
        if period_type in line.lower() and "projected to spend" in line.lower():
            return line.strip()
    return None

from typing import Optional
import pandas as pd
from datetime import datetime

def get_projected_spend(
    business_id: int,
    frequency: str = "yearly",
    year: Optional[int] = None,
    month: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    budget_data = get_budget_data(business_id)
    df = pd.DataFrame(budget_data.get("budgets", []))
    if df.empty:
        return {"projected_spend": 0.0, "currency": budget_data.get("currency", "NGN")}

    current_date = datetime.now()
    projected_spend = 0.0

    if frequency == "yearly" and year:
        # Projected spend for the given year
        days = 365
        projected_spend = df['amount_spent'].sum() + (df['daily_burn_rate'].sum() * days)
    elif frequency == "monthly" and month:
        # Projected spend for the given month of the current year
        days = 30
        projected_spend = df['amount_spent'].sum() + (df['daily_burn_rate'].sum() * days)
    elif frequency == "custom" and start_date and end_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end - start).days
        projected_spend = df['amount_spent'].sum() + (df['daily_burn_rate'].sum() * days)
    else:
        return {"error": "Invalid parameters."}

    return {
        "projected_spend": round(projected_spend, 2),
        "currency": budget_data.get("currency", "NGN"),
        "frequency": frequency,
        "period": {
            "year": year,
            "month": month,
            "start_date": start_date,
            "end_date": end_date
        }
    }

def get_department_insights(business_id: int, department_id: int):
    conn = mysql.connector.connect(
        host='13.247.208.85',
        port='3306',
        database='vgtechde_gopaddid',
        user='vgtechde_gopaddiv2',
        password='[VZNh-]E%{6q'
    )
    cursor = conn.cursor(dictionary=True)

    # Get department name
    cursor.execute(
        "SELECT name FROM departments WHERE id=%s AND business_id=%s", (department_id, business_id)
    )
    dept_row = cursor.fetchone()
    department_name = dept_row['name'] if dept_row else "Unknown"

    # Get expenses for department
    expense_query = """
        SELECT e.id as expense_id, e.code, e.department_id, e.business_id, e.approved_on, e.status,
               er.name as expense_name, er.amount, er.date, er.status as request_status, er.expense_category_id,
               ec.name as category_name
        FROM expenses e
        JOIN expense_requests er ON e.id = er.expense_id
        JOIN expense_categories ec ON er.expense_category_id = ec.id
        WHERE e.department_id = %s AND e.business_id = %s AND e.status = 'approved' AND er.status = 'approved' AND ec.status = 'active'
    """
    cursor.execute(expense_query, (department_id, business_id))
    expenses = cursor.fetchall()
    df = pd.DataFrame(expenses)

    insights = []
    if not df.empty:
        # Example insight 1: Top expense category
        top_cat = df.groupby('category_name')['amount'].sum().sort_values(ascending=False).head(1)
        if not top_cat.empty:
            cat_name = top_cat.index[0]
            cat_amt = top_cat.iloc[0]
            insights.append(f"Your highest spending category is {cat_name} with a total spend of {cat_amt:,.2f} NGN.")

        # Example insight 2: Fastest growing expense category (month-over-month)
        df['date'] = pd.to_datetime(df['date'])
        recent = df[df['date'] > datetime.now() - pd.DateOffset(months=6)]
        if not recent.empty:
            growth = recent.groupby(['category_name', pd.Grouper(key='date', freq='M')])['amount'].sum().unstack(fill_value=0)
            growth = growth.astype(float)
            if not growth.empty and growth.shape[1] > 1:
                growth_rate = (growth.iloc[:, -1] - growth.iloc[:, -2]) / (growth.iloc[:, -2] + 1e-6) * 100
                fastest = growth_rate.idxmax()
                rate = growth_rate.max()
                insights.append(f"Your spending on {fastest} increased by {rate:.1f}% last month.")

        # Example insight 3: Projected spend next month for top category
        if not top_cat.empty:
            avg_monthly = df[df['category_name'] == cat_name].groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().mean()
            projected = avg_monthly if avg_monthly else 0
            insights.append(f"If current trends continue, you may spend about {projected:,.2f} NGN on {cat_name} next month.")

        # Example insight 4: Total department spend vs. budget
        cursor.execute(
            "SELECT amount, amount_spent FROM budgets WHERE department_id=%s AND business_id=%s AND active=1",
            (department_id, business_id)
        )
        budget_row = cursor.fetchone()
        # Always fetch all results before next execute or closing
        while cursor.nextset():
            pass
        if budget_row:
            budget = budget_row['amount']
            spent = budget_row['amount_spent']
            utilization = (spent / budget * 100) if budget else 0
            insights.append(f"You have used {utilization:.1f}% of your department's budget ({spent:,.2f} NGN out of {budget:,.2f} NGN).")

        # Example insight 5: Largest single expense
        largest = df.loc[df['amount'].idxmax()]
        insights.append(f"Your largest single expense was {largest['expense_name']} ({largest['category_name']}) at {largest['amount']:,.2f} NGN on {largest['date'].strftime('%Y-%m-%d')}.")

        # Example insight 6: Most frequent expense category
        freq_cat = df['category_name'].mode()
        if not freq_cat.empty:
            insights.append(f"Your most frequent expense category is {freq_cat.iloc[0]}.")

        # --- LLM Insights Section ---
        if not df.empty:
            # Compose a summary for the LLM
            prompt = f"""
            You are a financial analyst. Given the following department expense and budget data, provide 2-3 concise, actionable insights (each as a single sentence with a number/figure if possible):

            Department: {department_name}
            Top expense category: {cat_name if 'cat_name' in locals() else 'N/A'}
            Total spend: {cat_amt:,.2f} NGN
            Budget utilization: {utilization:.1f}% ({spent:,.2f} NGN out of {budget:,.2f} NGN) if available

            Expense breakdown (category: total NGN):
            {df.groupby('category_name')['amount'].sum().to_dict()}

            Only reply with a JSON list of insights, e.g.:
            ["Insight 1...", "Insight 2...", "Insight 3..."]
            """

            try:
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a financial analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5
                )
                llm_output = response.choices[0].message.content
                import json
                llm_insights = json.loads(llm_output)
                if isinstance(llm_insights, list):
                    insights.extend(llm_insights)
            except Exception as e:
                insights.append(f"LLM insight generation failed: {e}")

    else:
        insights.append("No approved expenses found for this department.")

    cursor.close()
    conn.close()
    return {
        "department_insights": insights,
        "department_id": department_id,
        "department_name": department_name
    }

def get_department_projected_spend(
    business_id: int,
    department_id: int,
    frequency: str = "yearly",
    year: Optional[int] = None,
    month: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    # Get department budget data
    conn = mysql.connector.connect(
        host='13.247.208.85',
        port='3306',
        database='vgtechde_gopaddid',
        user='vgtechde_gopaddiv2',
        password='[VZNh-]E%{6q'
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT amount, amount_spent, start_date, end_date
        FROM budgets
        WHERE business_id = %s AND department_id = %s AND active = 1
        """,
        (business_id, department_id)
    )
    budgets = cursor.fetchall()
    cursor.close()
    conn.close()

    df = pd.DataFrame(budgets)
    if df.empty:
        return {"projected_spend": 0.0, "currency": "NGN"}

    # Convert to float and datetime
    for col in ['amount', 'amount_spent']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'])
    if 'end_date' in df.columns:
        df['end_date'] = pd.to_datetime(df['end_date'])

    # Calculate days elapsed and daily burn rate
    current_date = datetime.now()
    df['days_elapsed'] = (current_date - df['start_date']).dt.days
    df['daily_burn_rate'] = df.apply(
        lambda x: x['amount_spent'] / max(x['days_elapsed'], 1), axis=1
    )

    # Projected spend calculation
    if frequency == "yearly" and year:
        days = 365
    elif frequency == "monthly" and month:
        days = 30
    elif frequency == "custom" and start_date and end_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end - start).days
    else:
        return {"error": "Invalid parameters."}

    projected_spend = df['amount_spent'].sum() + (df['daily_burn_rate'].sum() * days)

    return {
        "projected_spend": round(projected_spend, 2),
        "currency": "NGN",
        "frequency": frequency,
        "period": {
            "year": year,
            "month": month,
            "start_date": start_date,
            "end_date": end_date
        },
        "department_id": department_id
    }

import mysql.connector
import pandas as pd
from datetime import datetime

def get_approval_trend_insights(business_id: int):
    conn = mysql.connector.connect(
        host='13.247.208.85',
        port='3306',
        database='vgtechde_gopaddid',
        user='vgtechde_gopaddiv2',
        password='[VZNh-]E%{6q'
    )
    cursor = conn.cursor(dictionary=True)

    # Query to get approval times for each expense request with category
    query = """
        SELECT 
            er.id as request_id,
            er.name as request_name,
            er.expense_category_id,
            ec.name as category_name,
            er.amount,
            er.date as request_date,
            e.approved_on as approved_on
        FROM expense_requests er
        JOIN expenses e ON er.expense_id = e.id
        JOIN expense_categories ec ON er.expense_category_id = ec.id
        WHERE er.business_id = %s
          AND e.business_id = %s
          AND ec.business_id = %s
          AND er.status = 'approved'
          AND e.status = 'approved'
          AND ec.status = 'active'
          AND e.approved_on IS NOT NULL
          AND er.date IS NOT NULL
    """
    cursor.execute(query, (business_id, business_id, business_id))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    df = pd.DataFrame(rows)
    insights = []

    if not df.empty:
        # Convert dates to datetime
        df['request_date'] = pd.to_datetime(df['request_date'])
        df['approved_on'] = pd.to_datetime(df['approved_on'])
        # Calculate approval time in days
        df['approval_time_days'] = (df['approved_on'] - df['request_date']).dt.days

        # Group by category and calculate average approval time
        cat_approval = df.groupby('category_name')['approval_time_days'].mean().sort_values()
        # Get top 3 fastest and slowest categories
        fastest = cat_approval.head(3)
        slowest = cat_approval.tail(3)

        # Insights for fastest categories
        for cat, days in fastest.items():
            insights.append(
                f"Based on past trends, requests for {cat} are likely to be approved within {days:.1f} days."
            )
        # Insights for slowest categories
        for cat, days in slowest.items():
            if cat not in fastest.index:  # Avoid duplicate if < 6 categories
                insights.append(
                    f"Requests for {cat} take longer to approve, averaging {days:.1f} days."
                )

        # Overall average
        overall = df['approval_time_days'].mean()
        insights.append(
            f"Across all categories, the average approval time for requests is {overall:.1f} days."
        )

        # LLM Insights Section
        if not df.empty:
            # Compose a summary for the LLM
            prompt = f"""
            You are a financial analyst. Given the following approval trend data for the organisation, provide 2-3 diverse, actionable insights (each as a single sentence with a number/figure if possible):

            Category approval times (category: avg days): {df.groupby('category_name')['approval_time_days'].mean().to_dict()}
            Overall average approval time: {df['approval_time_days'].mean():.1f} days

            Only reply with a JSON list of insights, e.g.:
            ["Insight 1...", "Insight 2...", "Insight 3..."]
            """

            try:
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a financial analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5
                )
                llm_output = response.choices[0].message.content
                import json
                llm_insights = json.loads(llm_output)
                if isinstance(llm_insights, list):
                    insights.extend(llm_insights)
            except Exception as e:
                insights.append(f"LLM insight generation failed: {e}")

    else:
        insights.append("No approved expense requests found for this organisation.")

    return {
        "approval_trend_insights": insights,
        "business_id": business_id
    }

def get_department_approval_trend_insights(business_id: int, department_id: int):
    conn = mysql.connector.connect(
        host='13.247.208.85',
        port='3306',
        database='vgtechde_gopaddid',
        user='vgtechde_gopaddiv2',
        password='[VZNh-]E%{6q'
    )
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT 
            er.id as request_id,
            er.name as request_name,
            er.expense_category_id,
            ec.name as category_name,
            er.amount,
            er.date as request_date,
            e.approved_on as approved_on
        FROM expense_requests er
        JOIN expenses e ON er.expense_id = e.id
        JOIN expense_categories ec ON er.expense_category_id = ec.id
        WHERE er.business_id = %s
          AND e.business_id = %s
          AND ec.business_id = %s
          AND e.department_id = %s
          AND er.status = 'approved'
          AND e.status = 'approved'
          AND ec.status = 'active'
          AND e.approved_on IS NOT NULL
          AND er.date IS NOT NULL
    """
    cursor.execute(query, (business_id, business_id, business_id, department_id))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    df = pd.DataFrame(rows)
    insights = []

    if not df.empty:
        df['request_date'] = pd.to_datetime(df['request_date'])
        df['approved_on'] = pd.to_datetime(df['approved_on'])
        df['approval_time_days'] = (df['approved_on'] - df['request_date']).dt.days

        cat_approval = df.groupby('category_name')['approval_time_days'].mean().sort_values()
        fastest = cat_approval.head(2)
        slowest = cat_approval.tail(2)

        for cat, days in fastest.items():
            insights.append(
                f"Based on past trends, requests for {cat} are likely to be approved within {days:.1f} days."
            )
        for cat, days in slowest.items():
            if cat not in fastest.index:
                insights.append(
                    f"Requests for {cat} take longer to approve, averaging {days:.1f} days."
                )
        overall = df['approval_time_days'].mean()
        insights.append(
            f"Across all categories in this department, the average approval time for requests is {overall:.1f} days."
        )

        # --- LLM Insights Section ---
        prompt = f"""
        You are a financial analyst. Given the following approval trend data for this department, provide 2-3 diverse, actionable insights (each as a single sentence with a number/figure if possible):

        Category approval times (category: avg days): {df.groupby('category_name')['approval_time_days'].mean().to_dict()}
        Overall average approval time: {df['approval_time_days'].mean():.1f} days

        Only reply with a JSON list of insights, e.g.:
        ["Insight 1...", "Insight 2...", "Insight 3..."]
        """

        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            llm_output = response.choices[0].message.content
            import json
            llm_insights = json.loads(llm_output)
            if isinstance(llm_insights, list):
                insights.extend(llm_insights)
        except Exception as e:
            insights.append(f"LLM insight generation failed: {e}")

    else:
        insights.append("No approved expense requests found for this department.")

    return {
        "approval_trend_insights": insights,
        "business_id": business_id,
        "department_id": department_id
    }

def main():
    # Get business_id from user input
    business_id = input("Enter the business ID: ")
    
    # Get budget data
    budget_data = get_budget_data(business_id)
    
    # Generate analysis
    analysis = generate_budget_analysis(budget_data)
    
    # Print the analysis
    print("\nBudget Projections:")
    print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()
