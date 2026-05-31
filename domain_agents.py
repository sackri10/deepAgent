"""
domain_agents.py
----------------
Orders and Sales domain DEEP AGENTS, each built with create_deep_agent().

These are fully autonomous agents with:
  - their own LLM + tools (scoped to their domain DB)
  - their own planning (write_todos built-in to deepagents)
  - their own context management (filesystem, summarisation)
  - structured output via Pydantic so the parent gets clean JSON

They are exposed as CompiledSubAgent objects so the orchestrator can
register them as sub-agents via the deepagents sub-agent delegation API.
"""

from pydantic import BaseModel, Field
from deepagents import create_deep_agent, CompiledSubAgent

from tools import ORDERS_TOOLS, SALES_TOOLS


# ─────────────────────────────────────────────────────────────
# Structured output schema — parent receives clean JSON
# ─────────────────────────────────────────────────────────────

class DomainQueryResult(BaseModel):
    """Structured result returned by a domain deep agent to the orchestrator."""
    domain: str = Field(description="Domain that answered: 'orders' or 'sales'")
    sql_query: str = Field(description="The SQL SELECT that produced the answer")
    key_findings: list[str] = Field(description="3-5 bullet-point findings")
    summary: str = Field(description="1-2 sentence synthesis of the findings")
    data_note: str = Field(
        default="",
        description="Any caveats, nulls, or data quality notes"
    )


# ─────────────────────────────────────────────────────────────
# ORDERS DEEP AGENT
# ─────────────────────────────────────────────────────────────

ORDERS_SYSTEM_PROMPT = """You are the Orders Domain Expert Agent.

Your mission: answer questions about customer orders, order status, delivery
tracking, and order-line details using SQL against the orders database.

## MANDATORY FIRST STEP — call write_todos() BEFORE anything else

You MUST call write_todos() as your very first action to create a visible plan:

  write_todos(todos=[
    {"task": "Get schema: call orders_get_schema()",           "done": false},
    {"task": "Write SQL query for: <restate the question>",    "done": false},
    {"task": "Execute SQL and verify results",                 "done": false},
    {"task": "Return structured answer",                       "done": false},
  ])

After each step, call write_todos() again marking completed items as done.

## Workflow
1. write_todos() with your plan  ← MANDATORY FIRST
2. Call orders_get_schema() to confirm table/column names.
3. Write a precise SELECT query — use JOINs when data spans multiple tables.
4. Call orders_execute_sql(query) to run it.
5. If result is empty or wrong, revise and retry once. Mark that step done.
6. write_todos() marking all steps done, then return your final answer.

## Output rules
- Return ONLY the key insights — no raw JSON dumps.
- Include the SQL query you used.
- 3-5 specific bullet findings with numbers.
- 1-2 sentence summary.
- domain: always "orders".
"""


def build_orders_agent(model) -> CompiledSubAgent:
    """
    Builds the Orders deep agent and wraps it as a CompiledSubAgent
    so it can be passed into the orchestrator's subagents=[...] list.
    """
    orders_agent = create_deep_agent(
        model=model,
        tools=ORDERS_TOOLS,
        system_prompt=ORDERS_SYSTEM_PROMPT,
        name="orders-agent",
        # response_format makes the parent receive DomainQueryResult as JSON
        # NOTE: set this on the CompiledSubAgent wrapper, not here
    )

    return CompiledSubAgent(
        name="orders-agent",
        description=(
            "Specialist for customer orders, order status, delivery tracking, "
            "order totals, and order-line details. Use when the question involves "
            "orders, customers, shipments, order counts, or per-customer spend."
        ),
        runnable=orders_agent,
    )


# ─────────────────────────────────────────────────────────────
# SALES DEEP AGENT
# ─────────────────────────────────────────────────────────────

SALES_SYSTEM_PROMPT = """You are the Sales Domain Expert Agent.

Your mission: answer questions about sales revenue, product performance,
sales rep quotas, and regional targets using SQL against the sales database.

## MANDATORY FIRST STEP — call write_todos() BEFORE anything else

You MUST call write_todos() as your very first action to create a visible plan:

  write_todos(todos=[
    {"task": "Get schema: call sales_get_schema()",            "done": false},
    {"task": "Write SQL query for: <restate the question>",    "done": false},
    {"task": "Execute SQL and verify results",                 "done": false},
    {"task": "Return structured answer",                       "done": false},
  ])

After each step, call write_todos() again marking completed items as done.

## Workflow
1. write_todos() with your plan  ← MANDATORY FIRST
2. Call sales_get_schema() to confirm table/column names.
3. Write a precise SELECT — use JOINs and aggregations (SUM, AVG, GROUP BY).
4. Call sales_execute_sql(query) to run it.
5. If result is empty or wrong, revise and retry once. Mark that step done.
6. write_todos() marking all steps done, then return your final answer.

## Output rules
- Return ONLY the key insights — no raw JSON dumps.
- Include the SQL query you used.
- 3-5 specific bullet findings with numbers.
- 1-2 sentence summary.
- domain: always "sales".
"""


def build_sales_agent(model) -> CompiledSubAgent:
    """
    Builds the Sales deep agent and wraps it as a CompiledSubAgent.
    """
    sales_agent = create_deep_agent(
        model=model,
        tools=SALES_TOOLS,
        system_prompt=SALES_SYSTEM_PROMPT,
        name="sales-agent",
    )

    return CompiledSubAgent(
        name="sales-agent",
        description=(
            "Specialist for sales revenue, product performance, sales rep quotas, "
            "regional sales targets, and margin analysis. Use when the question "
            "involves revenue, sales reps, products sold, regions, or targets."
        ),
        runnable=sales_agent,
    )