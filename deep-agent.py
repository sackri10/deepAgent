"""
Text-to-SQL Deep Agent with Streaming
======================================
Demonstrates LangChain DeepAgents framework with:
  - Orchestrator (main) agent using write_todos to plan
  - Sales subagent for sales-domain queries
  - Orders subagent for orders-domain queries
  - Full streaming: todos, tokens, tool calls, subagent lifecycle

Streaming uses v2 format with combined ["updates", "messages"] modes
so we can see:
  1. write_todos calls from the orchestrator  ->  plan becomes visible
  2. task tool calls                           ->  which subagent is delegated to
  3. token-by-token LLM output from each agent
  4. SQL tool calls + results inside subagents

Setup
-----
pip install deepagents langchain langchain-anthropic

Export:
  export ANTHROPIC_API_KEY=sk-ant-...

Run:
  python text_to_sql_deep_agent.py
"""

import asyncio
import json
import os
import sys
from datetime import date
from typing import Any

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

import os
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Fake in-memory database (replace with real SQLAlchemy / pyodbc session)
# ---------------------------------------------------------------------------

SALES_DATA = [
    {"sale_id": 1, "product": "Widget A", "amount": 1200.00, "region": "EMEA",  "sale_date": "2025-01-15"},
    {"sale_id": 2, "product": "Widget B", "amount":  850.50, "region": "APAC",  "sale_date": "2025-02-10"},
    {"sale_id": 3, "product": "Gadget X", "amount": 3400.00, "region": "NA",    "sale_date": "2025-03-01"},
    {"sale_id": 4, "product": "Widget A", "amount": 1100.00, "region": "EMEA",  "sale_date": "2025-03-20"},
    {"sale_id": 5, "product": "Gadget Y", "amount": 2200.00, "region": "APAC",  "sale_date": "2025-04-05"},
]

ORDERS_DATA = [
    {"order_id": 101, "customer": "Acme Corp",    "status": "shipped",    "total": 1200.00, "order_date": "2025-01-14"},
    {"order_id": 102, "customer": "Globex Ltd",   "status": "processing", "total":  850.50, "order_date": "2025-02-09"},
    {"order_id": 103, "customer": "Initech",      "status": "delivered",  "total": 3400.00, "order_date": "2025-03-01"},
    {"order_id": 104, "customer": "Umbrella Inc", "status": "cancelled",  "total":  320.00, "order_date": "2025-03-18"},
    {"order_id": 105, "customer": "Acme Corp",    "status": "shipped",    "total": 2200.00, "order_date": "2025-04-04"},
]


# ---------------------------------------------------------------------------
# SQL execution simulation (replace body with real DB call)
# ---------------------------------------------------------------------------

def _execute_sql(sql: str, domain: str) -> list[dict]:
    """Very naive in-memory SQL executor for the POC."""
    sql_lower = sql.lower().strip()
    dataset = SALES_DATA if domain == "sales" else ORDERS_DATA

    # SELECT *
    if "select *" in sql_lower or "select all" in sql_lower:
        return dataset

    # WHERE region = 'EMEA'
    if "where" in sql_lower and "region" in sql_lower:
        for region in ["EMEA", "APAC", "NA"]:
            if region.lower() in sql_lower:
                return [r for r in dataset if r.get("region") == region]

    # WHERE status = '...'
    if "where" in sql_lower and "status" in sql_lower:
        for status in ["shipped", "processing", "delivered", "cancelled"]:
            if status in sql_lower:
                return [r for r in dataset if r.get("status") == status]

    # GROUP BY / SUM / aggregate — return summary
    if "sum" in sql_lower or "group by" in sql_lower or "aggregate" in sql_lower:
        if domain == "sales":
            by_product: dict[str, float] = {}
            for row in dataset:
                by_product[row["product"]] = by_product.get(row["product"], 0) + row["amount"]
            return [{"product": k, "total_amount": v} for k, v in by_product.items()]
        else:
            by_status: dict[str, int] = {}
            for row in dataset:
                by_status[row["status"]] = by_status.get(row["status"], 0) + 1
            return [{"status": k, "count": v} for k, v in by_status.items()]

    # Default: return all
    return dataset


# ---------------------------------------------------------------------------
# Tools for subagents
# ---------------------------------------------------------------------------

@tool
def run_sales_sql(sql: str) -> str:
    """Execute a SQL query against the Sales database.

    Use standard SQL syntax. Available table: sales(sale_id, product, amount, region, sale_date).
    Returns results as JSON array.
    """
    try:
        rows = _execute_sql(sql, "sales")
        return json.dumps(rows, indent=2)
    except Exception as exc:
        return f"SQL Error: {exc}"


@tool
def run_orders_sql(sql: str) -> str:
    """Execute a SQL query against the Orders database.

    Use standard SQL syntax. Available table: orders(order_id, customer, status, total, order_date).
    Returns results as JSON array.
    """
    try:
        rows = _execute_sql(sql, "orders")
        return json.dumps(rows, indent=2)
    except Exception as exc:
        return f"SQL Error: {exc}"


# ---------------------------------------------------------------------------
# Subagent definitions
# ---------------------------------------------------------------------------

SALES_SUBAGENT = {
    "name": "sales-agent",
    "description": (
        "Handles all queries related to the Sales domain: "
        "revenue, product performance, regional breakdowns, sales trends."
    ),
    "system_prompt": (
        "You are an expert Sales Data Analyst. "
        "Your ONLY data source is the run_sales_sql tool. "
        "Schema: sales(sale_id INT, product VARCHAR, amount DECIMAL, region VARCHAR, sale_date DATE). "
        "ALWAYS call run_sales_sql with a valid SQL query before answering. "
        "Present results in a clear table or bullet format. "
        "Do NOT make up numbers."
    ),
    "tools": [run_sales_sql],
}

ORDERS_SUBAGENT = {
    "name": "orders-agent",
    "description": (
        "Handles all queries related to the Orders domain: "
        "order status, customer orders, fulfilment rates, cancellations."
    ),
    "system_prompt": (
        "You are an expert Orders Data Analyst. "
        "Your ONLY data source is the run_orders_sql tool. "
        "Schema: orders(order_id INT, customer VARCHAR, status VARCHAR, total DECIMAL, order_date DATE). "
        "ALWAYS call run_orders_sql with a valid SQL query before answering. "
        "Present results in a clear table or bullet format. "
        "Do NOT make up numbers."
    ),
    "tools": [run_orders_sql],
}


# ---------------------------------------------------------------------------
# Orchestrator system prompt — explicitly requires write_todos
# ---------------------------------------------------------------------------

ORCHESTRATOR_PROMPT = f"""
You are a Business Intelligence Orchestrator powered by LangChain DeepAgents.
Today's date is {date.today().isoformat()}.

## Responsibilities
1. Understand the user's natural-language business question.
2. **ALWAYS call write_todos first** to create a structured plan before doing any work.
   Your todos must list:
   - Which sub-agent(s) to delegate to (sales-agent, orders-agent, or both)
   - What specific question each sub-agent should answer
   - How you will synthesise the final answer
3. Delegate to the correct sub-agent(s) using the `task` tool:
   - Use **sales-agent** for revenue, product, and regional questions.
   - Use **orders-agent** for order status, fulfilment, customer queries.
   - Use BOTH in parallel when the question spans both domains.
4. Synthesise sub-agent results into a coherent final answer with insights.

## Rules
- NEVER query the database yourself — always delegate via task tool.
- ALWAYS write_todos before delegating.
- Update todos to 'completed' after each step.
- If unsure which domain, delegate to both agents.
"""


# ---------------------------------------------------------------------------
# Stream handler — prints plan, tokens, tool calls, subagent lifecycle
# ---------------------------------------------------------------------------

COLORS = {
    "reset":    "\033[0m",
    "bold":     "\033[1m",
    "cyan":     "\033[36m",
    "green":    "\033[32m",
    "yellow":   "\033[33m",
    "magenta":  "\033[35m",
    "blue":     "\033[34m",
    "red":      "\033[31m",
    "dim":      "\033[2m",
}

def c(color: str, text: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"



def _nl(state: dict) -> None:
    """Ensure we're on a fresh line."""
    if state["mid_line"]:
        print()
        state["mid_line"] = False


def _print_todos(args: dict) -> None:
    todos_list = args.get("todos", args.get("items", []))
    print()
    print(c("bold", "╔══ 📋 ORCHESTRATOR PLAN (write_todos) ══╗"))
    for i, todo in enumerate(todos_list, 1):
        if isinstance(todo, dict):
            status  = todo.get("status", "pending")
            content = todo.get("content", todo.get("task", str(todo)))
            icon    = "✅" if status == "completed" else "🔄" if status == "in_progress" else "⬜"
            print(f"  {icon} {i}. {content}")
        else:
            print(f"  ⬜ {i}. {todo}")
    print(c("bold", "╚════════════════════════════════════════╝"))
    print()


def handle_stream_chunk(chunk: dict[str, Any], state: dict) -> None:
    """
    state keys:
      mid_line                 bool  – printed chars without trailing newline
      orchestrator_sources     dict  – track which orchestrator phases have printed headers
      subagent_sources         set   – subagent namespaces that have printed thinking headers
      subagents_complete       bool  – True when all subagent results have been returned
      orchestrator_has_synth   bool  – track if final answer header has been printed
    """
    ns:         tuple = chunk.get("ns", ())
    chunk_type: str   = chunk.get("type", "")
    data:       Any   = chunk.get("data", {})

    is_subagent  = any(s.startswith("tools:") for s in ns)
    subagent_seg = next((s for s in ns if s.startswith("tools:")), "") if is_subagent else ""

    # ------------------------------------------------------------------ UPDATES
    if chunk_type == "updates":
        if not isinstance(data, dict):
            return

        for node_name, node_data in data.items():
            if not isinstance(node_data, dict):
                continue
            messages = node_data.get("messages", [])

            for msg in messages:
                msg_type   = getattr(msg, "type", "")
                tool_calls = getattr(msg, "tool_calls", [])

                # orchestrator tool calls: write_todos + task
                if msg_type == "ai" and not is_subagent:
                    for tc in tool_calls:
                        name = tc.get("name", "")
                        args = tc.get("args", {})
                        if name == "write_todos":
                            _nl(state)
                            _print_todos(args)
                        elif name == "task":
                            _nl(state)
                            sub_type    = args.get("subagent_type", args.get("name", "unknown"))
                            description = args.get("description", args.get("task", ""))[:120]
                            print(c("yellow", f"🚀 Delegating to [{sub_type}]: {description}"))
                    
                    # Orchestrator's final synthesis message (no tool calls after subagents complete)
                    if not tool_calls and state["subagents_complete"]:
                        msg_content = getattr(msg, "content", "")
                        if msg_content and not state["orchestrator_has_synth"]:
                            _nl(state)
                            print(f"\n{c('cyan', '─── Orchestrator final answer ───')}")
                            
                            # Extract text from content blocks if it's a list
                            if isinstance(msg_content, list):
                                for block in msg_content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        print(block.get("text", ""))
                                    elif isinstance(block, str):
                                        print(block)
                            else:
                                print(msg_content)
                            
                            state["orchestrator_has_synth"] = True

                # subagent SQL tool calls
                if msg_type == "ai" and is_subagent:
                    for tc in tool_calls:
                        if tc.get("name") in ("run_sales_sql", "run_orders_sql"):
                            _nl(state)
                            sql = tc.get("args", {}).get("sql", "")
                            print(c("green", f"  🗄  SQL → {sql}"))

                # subagent SQL results
                if msg_type == "tool" and is_subagent:
                    if getattr(msg, "name", "") in ("run_sales_sql", "run_orders_sql"):
                        _nl(state)
                        print(c("dim", f"  📊 Result: {str(msg.content)[:300]}..."))

            # subagent finished — result returned to orchestrator
            if not ns and node_name == "tools":
                for msg in messages:
                    if getattr(msg, "type", None) == "tool":
                        _nl(state)
                        print(c("green", f"✅ Subagent done: {msg.tool_call_id[:20]}..."))
            
            # Track when subagents have completed
            if not ns and node_name == "tools" and messages:
                state["subagents_complete"] = True

    # ------------------------------------------------------------------ MESSAGES
    elif chunk_type == "messages":
        if not isinstance(data, (list, tuple)) or len(data) < 1:
            return

        token      = data[0]
        token_type = getattr(token, "type", "")
        if token_type != "ai":
            return

        content          = getattr(token, "content", "") or ""
        tool_call_chunks = getattr(token, "tool_call_chunks", None) or []

        if not content and not tool_call_chunks:
            return

        # Handle subagent tokens
        if is_subagent:
            source_key = subagent_seg
            if source_key not in state["subagent_sources"]:
                _nl(state)
                print(f"\n{c('magenta', f'─── {subagent_seg} thinking ───')}")
                state["subagent_sources"].add(source_key)
            
            if content and not tool_call_chunks:
                print(content, end="", flush=True)
                state["mid_line"] = not content.endswith("\n")
        
        # Handle orchestrator tokens
        else:
            # During initial delegation phase: print thinking header
            if not state["subagents_complete"]:
                if "thinking" not in state["orchestrator_sources"]:
                    _nl(state)
                    print(f"\n{c('cyan', '─── Orchestrator thinking ───')}")
                    state["orchestrator_sources"]["thinking"] = True
            else:
                # After subagents complete, orchestrator enters synthesis phase
                if not state["orchestrator_has_synth"]:
                    _nl(state)
                    print(f"\n{c('cyan', '─── Orchestrator final answer ───')}")
                    state["orchestrator_has_synth"] = True
            
            if content and not tool_call_chunks:
                print(content, end="", flush=True)
                state["mid_line"] = not content.endswith("\n")


# ---------------------------------------------------------------------------
# Main: create agent and run streaming query
# ---------------------------------------------------------------------------

def create_agent():
    model = init_chat_model(
        model="anthropic:claude-sonnet-4-5",
        temperature=0,
    )

    agent = create_deep_agent(
        model=model,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=[SALES_SUBAGENT, ORDERS_SUBAGENT],
    )
    return agent


def stream_query(agent, query: str) -> None:
    print("\n" + "═" * 60)
    print(c("bold", f"USER QUERY: {query}"))
    print("═" * 60)

    stream_state = {
        "mid_line":              False,
        "orchestrator_sources":  {},      # tracks orchestrator phases (thinking, synthesis)
        "subagent_sources":      set(),   # tracks subagent namespaces
        "subagents_complete":    False,   # True when all subagent tool results returned
        "orchestrator_has_synth": False,  # tracks if final answer header has been printed
    }

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode=["updates", "messages"],
        subgraphs=True,
        version="v2",
    ):
        handle_stream_chunk(chunk, stream_state)

    if stream_state["mid_line"]:
        print()

    print("\n" + "═" * 60 + "\n")


# ---------------------------------------------------------------------------
# Demo queries
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "What is the total sales revenue by product? Also show me the count of orders by status.",
    "Show me all sales in the EMEA region.",
    "Which customers have shipped orders and what are their totals?",
]


def main():
    print(c("bold", "\n🤖 Text-to-SQL Deep Agent — Streaming POC"))
    print(c("dim", "Uses LangChain DeepAgents + Anthropic Claude Sonnet"))
    print(c("dim", "Streaming: todos plan, tokens, SQL tool calls, subagent lifecycle\n"))

    agent = create_agent()

    # Run one or all demo queries
    queries_to_run = [DEMO_QUERIES[0]]  # Change index or use DEMO_QUERIES for all

    for q in queries_to_run:
        stream_query(agent, q)


if __name__ == "__main__":
    main()