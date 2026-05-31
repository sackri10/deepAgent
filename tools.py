"""
tools.py
--------
SQL executor tools for each domain database.
Each tool is a plain Python function decorated with @tool.
These are passed into create_deep_agent(tools=[...]) for each domain sub-agent.

Design: one tool set per domain → the Orders sub-agent physically cannot
query the sales DB, and vice versa. Isolation by construction, not by policy.
"""

import sqlite3, json, os
from langchain_core.tools import tool

BASE     = os.path.join(os.path.dirname(__file__), "data")
ORDERS_DB = os.path.join(BASE, "orders.db")
SALES_DB  = os.path.join(BASE, "sales.db")

# ─────────────────────────────────────────────────────────────
# ORDERS tools
# ─────────────────────────────────────────────────────────────

@tool
def orders_get_schema() -> str:
    """
    Return the full schema of the orders database.
    Call this first before writing any SQL to confirm table and column names.
    """
    return """
Database: orders.db

Tables:
  customers  (customer_id PK, name, email, country, signup_date DATE)
  orders     (order_id PK, customer_id FK→customers, order_date DATE,
               status TEXT ['pending'|'confirmed'|'shipped'|'delivered'|'cancelled'],
               total_amount REAL)
  order_items (item_id PK, order_id FK→orders, product_id INT,
                product_name TEXT, quantity INT, unit_price REAL)

Relationships:
  customers  1──< orders 1──< order_items
"""


@tool
def orders_execute_sql(query: str) -> str:
    """
    Execute a SELECT query against the orders database.
    Returns rows as a JSON array. Returns an error string on failure.
    Only SELECT is allowed — never mutate data.

    Args:
        query: A valid SQLite SELECT statement.
    """
    query = query.strip()
    if not query.upper().startswith("SELECT"):
        return "ERROR: Only SELECT queries are allowed."
    try:
        con = sqlite3.connect(ORDERS_DB)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(query)
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        if not rows:
            return "Query succeeded but returned 0 rows."
        return json.dumps(rows, indent=2, default=str)
    except Exception as e:
        return f"SQL ERROR: {e}"


# ─────────────────────────────────────────────────────────────
# SALES tools
# ─────────────────────────────────────────────────────────────

@tool
def sales_get_schema() -> str:
    """
    Return the full schema of the sales database.
    Call this first before writing any SQL to confirm table and column names.
    """
    return """
Database: sales.db

Tables:
  products           (product_id PK, product_name, category,
                      unit_price REAL, cost_price REAL)
  sales_reps         (rep_id PK, name, region, quota REAL)
  sales_transactions (txn_id PK, rep_id FK→sales_reps,
                      product_id FK→products, txn_date DATE,
                      quantity INT, revenue REAL, region TEXT)
  sales_targets      (target_id PK, region, year INT, quarter INT,
                      target_amount REAL)

Regions: 'North America' | 'Europe' | 'Asia Pacific'
"""


@tool
def sales_execute_sql(query: str) -> str:
    """
    Execute a SELECT query against the sales database.
    Returns rows as a JSON array. Returns an error string on failure.
    Only SELECT is allowed — never mutate data.

    Args:
        query: A valid SQLite SELECT statement.
    """
    query = query.strip()
    if not query.upper().startswith("SELECT"):
        return "ERROR: Only SELECT queries are allowed."
    try:
        con = sqlite3.connect(SALES_DB)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(query)
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        if not rows:
            return "Query succeeded but returned 0 rows."
        return json.dumps(rows, indent=2, default=str)
    except Exception as e:
        return f"SQL ERROR: {e}"


# Convenience exports
ORDERS_TOOLS = [orders_get_schema, orders_execute_sql]
SALES_TOOLS  = [sales_get_schema,  sales_execute_sql]
