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
from langchain_anthropic import ChatAnthropic
from domain_agents import build_orders_agent, build_sales_agent
from llm import llm_aws,model,sub_agent_model

# ─────────────────────────────────────────────────────────────
# Orchestrator system prompt
# ─────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """You are a Text-to-SQL Orchestrator Agent.

Your job: translate natural-language business questions into structured answers
by coordinating specialist sub-agents via the task() tool.

You have TWO specialist deep agents available:

  orders-agent
    Domain: customer orders, order status, delivery, order totals, per-customer spend
    When to use: question mentions orders, customers, shipments, order counts

  sales-agent
    Domain: product revenue, sales rep performance, regional quotas, product margins
    When to use: question mentions revenue, sales, products sold, reps, regions, targets

Decision rules:
  - If the question spans BOTH domains → call task() for EACH agent with a focused sub-question.
  - If it's clearly one domain → call task() for that agent only.
  - Always reformulate the sub-question to be specific to that domain's data.

Delegation format (call task() like this):
  task(name="orders-agent", task="<focused orders question>")
  task(name="sales-agent",  task="<focused sales question>")

After you receive results from all sub-agents:
  - Synthesise them into a single, coherent, well-structured answer.
  - Use markdown: bold key numbers, bullet points for lists.
  - Cross-reference data between domains where meaningful.
  - Do NOT expose raw SQL or JSON in the final answer.
  - Do NOT ask follow-up questions — give a complete answer.

IMPORTANT:
  - Always delegate — do NOT attempt SQL yourself.
  - Be explicit in sub-questions: include time ranges, grouping criteria, and
    any filtering the user implied.
  - If a sub-agent returns an error, note it and synthesise from the rest.
"""


# ─────────────────────────────────────────────────────────────
# Build the orchestrator
# ─────────────────────────────────────────────────────────────

def build_orchestrator(model: str = "anthropic:claude-sonnet-4-6") -> object:
    """
    Assembles the full multi-agent system:
      Text-to-SQL Orchestrator (deep agent)
        ├── orders-agent  (CompiledSubAgent wrapping a deep agent)
        └── sales-agent   (CompiledSubAgent wrapping a deep agent)

    The orchestrator uses a stronger model (gpt-4o) for coordination;
    sub-agents use a cheaper model (gpt-4o-mini) for SQL execution.
    """ 
   
    orders_subagent = build_orders_agent(model=sub_agent_model)
    sales_subagent  = build_sales_agent(model=sub_agent_model)

    orchestrator = create_deep_agent(
        model=llm_aws,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        name="text-to-sql-orchestrator",
        subagents=[
            orders_subagent,
            sales_subagent,
        ]
    )

    return orchestrator


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def ask_streaming_tokens(question: str):
    """
    Token-by-token streaming. Shows raw LLM output as it generates.
    """
    orchestrator = build_orchestrator()

    print(f"\nQ: {question}\n")

    for chunk, metadata in orchestrator.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="messages",
    ):
        # metadata contains lc_agent_name to identify which agent is speaking
        agent_name = metadata.get("metadata", {}).get("lc_agent_name", "unknown")
        content = getattr(chunk, "content", "")

        if content and not getattr(chunk, "tool_calls", None):
            # Label changes only when agent changes
            print(content, end="", flush=True)
                
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
