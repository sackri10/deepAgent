"""
orchestrator.py
---------------
The Text-to-SQL Orchestrator — itself a deep agent built with create_deep_agent().

It receives natural-language questions and decides which domain sub-agents
to delegate to via the built-in task() tool that deepagents injects automatically.

The orchestrator does NOT execute SQL itself. It:
  1. Decomposes the user question into domain-scoped sub-tasks
  2. Delegates each sub-task to the appropriate deep agent via task()
  3. Synthesises the structured JSON results into a final NL answer

Key deepagents concepts used here:
  - create_deep_agent()   : builds the orchestrator with sub-agent awareness
  - CompiledSubAgent      : wraps each domain deep-agent as a delegate
  - subagents=[...]       : registers delegates; deepagents injects task() tool
  - system_prompt         : instructs the orchestrator HOW to decompose + delegate
"""

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage

from domain_agents import build_orders_agent, build_sales_agent
from llm import model,sub_agent_model
from llm import llm_aws

# ─────────────────────────────────────────────────────────────
# Orchestrator system prompt
# ─────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """You are a Text-to-SQL Orchestrator Agent.

Your job: translate natural-language business questions into structured answers
by coordinating specialist sub-agents via the task() tool.

## MANDATORY FIRST STEP — YOU MUST ALWAYS DO THIS BEFORE ANYTHING ELSE

Before calling task() or doing ANY other work, you MUST call write_todos() to
create a visible plan. No exceptions. Format:

  write_todos(todos=[
    {"task": "Analyse question: decide which domain(s) are needed", "done": false},
    {"task": "Delegate to orders-agent: <specific sub-question>",   "done": false},
    {"task": "Delegate to sales-agent: <specific sub-question>",    "done": false},
    {"task": "Synthesise results into final answer",                "done": false},
  ])

Only include the domains actually needed. After write_todos(), proceed with
the plan step by step, calling write_todos() again to mark items done as you go.

## Available sub-agents

  orders-agent
    Domain: customer orders, order status, delivery, order totals, per-customer spend
    When to use: question mentions orders, customers, shipments, order counts

  sales-agent
    Domain: product revenue, sales rep performance, regional quotas, product margins
    When to use: question mentions revenue, sales, products sold, reps, regions, targets

## Delegation rules
  - Cross-domain question → call task() for EACH agent with a focused sub-question.
  - Single domain → call task() for that agent only.
  - Always reformulate sub-questions to be specific to that domain's data.
  - Always delegate — do NOT attempt SQL yourself.

## After receiving sub-agent results
  - Mark delegation todos as done with write_todos().
  - Synthesise into a single, coherent, well-structured markdown answer.
  - Bold key numbers, use bullet points for lists.
  - Cross-reference data between domains where meaningful.
  - Do NOT expose raw SQL or JSON in the final answer.
  - Do NOT ask follow-up questions — give a complete answer.
  - Mark the synthesis todo as done.
"""


# ─────────────────────────────────────────────────────────────
# Build the orchestrator
# ─────────────────────────────────────────────────────────────

def build_orchestrator(model1: str = "openai:gpt-4o") -> object:
    """
    Assembles the full multi-agent system:
      Text-to-SQL Orchestrator (deep agent)
        ├── orders-agent  (CompiledSubAgent wrapping a deep agent)
        └── sales-agent   (CompiledSubAgent wrapping a deep agent)

    The orchestrator uses a stronger model (gpt-4o) for coordination;
    sub-agents use a cheaper model (gpt-4o-mini) for SQL execution.
    """ 

    orders_subagent = build_orders_agent(sub_agent_model)
    sales_subagent  = build_sales_agent(sub_agent_model)

    orchestrator = create_deep_agent(
        model=model,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        name="text-to-sql-orchestrator",
        subagents=[
            orders_subagent,
            sales_subagent,
        ],
        # No SQL tools on the orchestrator — it only coordinates
        tools=[],
    )

    return orchestrator


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def ask(question: str, model: str = "openai:gpt-4o") -> str:
    """
    Ask a natural-language question.
    The orchestrator routes to the right domain agent(s) and returns
    a synthesised markdown answer.

    Args:
        question: Any business question about orders or sales (or both).
        model:    LLM for the orchestrator. Sub-agents always use gpt-4o-mini.

    Returns:
        Final synthesised answer as a markdown string.
    """
    orchestrator = build_orchestrator(model=model)

    result = orchestrator.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # deepagents returns state with 'messages'; last AIMessage is the answer
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            # Skip tool call messages (they have tool_calls, not plain content)
            if not getattr(msg, "tool_calls", None):
                return msg.content

    return "No answer produced."


# ─────────────────────────────────────────────────────────────
# Async variant (for FastAPI / async callers)
# ─────────────────────────────────────────────────────────────

async def ask_async(question: str, model: str = "openai:gpt-4o") -> str:
    """Async version of ask(). Use with `await` in async contexts."""
    orchestrator = build_orchestrator(model=model)

    result = await orchestrator.ainvoke({
        "messages": [HumanMessage(content=question)]
    })

    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            if not getattr(msg, "tool_calls", None):
                return msg.content

    return "No answer produced."