# Multi-Agent Text-to-SQL — built on `deepagents`

A hierarchical multi-deep-agent system where each layer is a genuine
`create_deep_agent()` instance from the [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents) framework.

---

## Architecture

```
User question
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  Text-to-SQL Orchestrator  (create_deep_agent)               │
│                                                              │
│  System prompt: decompose → delegate → synthesise            │
│  Tools: none (coordinates only)                              │
│  Sub-agents: injects built-in task() tool automatically      │
└──────────────────┬──────────────────────┬────────────────────┘
                   │  task()              │  task()
                   ▼                      ▼
    ┌──────────────────────┐  ┌──────────────────────┐
    │  Orders Deep Agent   │  │  Sales Deep Agent    │
    │  (create_deep_agent) │  │  (create_deep_agent) │
    │                      │  │                      │
    │  Tools:              │  │  Tools:              │
    │  · orders_get_schema │  │  · sales_get_schema  │
    │  · orders_execute_sql│  │  · sales_execute_sql │
    │                      │  │                      │
    │  Built-in harness:   │  │  Built-in harness:   │
    │  · planning          │  │  · planning          │
    │  · context mgmt      │  │  · context mgmt      │
    │  · filesystem        │  │  · filesystem        │
    └──────────┬───────────┘  └──────────┬───────────┘
               │  DomainQueryResult JSON  │
               └──────────┬───────────────┘
                           ▼
                  Orchestrator synthesises
                  → Final NL answer (markdown)
```

### Why each layer is a real deep agent

| Layer | Framework construct | What it adds |
|---|---|---|
| Orchestrator | `create_deep_agent(subagents=[...])` | Built-in `task()` tool for delegation, planning, context isolation |
| Orders agent | `create_deep_agent(tools=ORDERS_TOOLS)` | Planning loop, schema introspection, SQL retry, context offloading |
| Sales agent | `create_deep_agent(tools=SALES_TOOLS)` | Same as orders, scoped to sales DB |

---

## Key `deepagents` patterns used

### 1. Sub-agent registration
```python
orders_subagent = CompiledSubAgent(
    name="orders-agent",
    description="...",          # orchestrator uses this to decide when to delegate
    runnable=orders_deep_agent  # a compiled create_deep_agent() graph
)

orchestrator = create_deep_agent(
    model="openai:gpt-4o",
    subagents=[orders_subagent, sales_subagent],  # injects task() tool automatically
    system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
)
```

### 2. Delegation via `task()` (built-in, no manual wiring)
The orchestrator LLM calls:
```
task(name="orders-agent", task="Which customers have spent over $2000 in delivered orders?")
task(name="sales-agent",  task="What is the total revenue per region in Q1 2024?")
```
Each sub-agent runs with its own isolated context — the orchestrator never
sees the intermediate SQL calls, only the final structured result.

### 3. Context isolation (why this beats custom LangGraph for this use case)
Each `task()` call spawns the sub-agent in its own context window. The
20+ SQL tool calls inside the Orders agent don't pollute the orchestrator's
context — only the summary comes back. This is what `deepagents` calls
**context quarantine**.

### 4. `CompiledSubAgent` for deep-agent-as-subagent
```python
# The sub-agent is itself a full deep agent:
orders_agent_graph = create_deep_agent(
    model="openai:gpt-4o-mini",
    tools=ORDERS_TOOLS,
    system_prompt=ORDERS_SYSTEM_PROMPT,
)

# Wrap it so the orchestrator can register it:
CompiledSubAgent(
    name="orders-agent",
    description="...",
    runnable=orders_agent_graph,
)
```

---

## Project structure

```
multi_agent_text2sql/
├── main.py           ← Entry point; run this
├── orchestrator.py   ← Text-to-SQL orchestrator (create_deep_agent + subagents)
├── domain_agents.py  ← Orders and Sales deep agents (create_deep_agent)
├── tools.py          ← SQL executor + schema tools (one set per domain)
├── db_setup.py       ← SQLite seed data
├── requirements.txt
└── data/
    ├── orders.db     ← Created at runtime
    └── sales.db      ← Created at runtime
```

---

## Setup

```bash
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY=sk-...
# Or create .env: OPENAI_API_KEY=sk-...

python main.py
```

### Using Claude (Anthropic) instead

```python
# In orchestrator.py
orchestrator = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",   # provider:model format
    ...
)

# In domain_agents.py
orders_agent = create_deep_agent(
    model="anthropic:claude-haiku-4-5",    # cheaper model for SQL work
    ...
)
```

`deepagents` is model-agnostic — any LangChain chat model that supports
tool calling works. The `"provider:model"` string format is shorthand
for `init_chat_model("model", model_provider="provider")`.

---

## Adding a third domain (e.g. Inventory)

1. **`tools.py`** — add `inventory_get_schema()` and `inventory_execute_sql()` tools
2. **`domain_agents.py`** — add `build_inventory_agent()` returning a `CompiledSubAgent`
3. **`orchestrator.py`** — add `inventory_subagent` to the `subagents=[...]` list
   and update `ORCHESTRATOR_SYSTEM_PROMPT` to describe the new domain

The orchestrator LLM will automatically learn to route to `inventory-agent`
based on the description and system prompt — no graph rewiring needed.

---

## Deepagents vs raw LangGraph

| Concern | Raw LangGraph | deepagents |
|---|---|---|
| Sub-agent wiring | Manual `Send()`, reducers, node registration | `subagents=[...]`, framework injects `task()` |
| Context isolation | Manual (you manage message history per node) | Built-in context quarantine per `task()` call |
| Planning | Build your own | `write_todos` built-in |
| Retry / resilience | Build your own | Built-in to harness |
| Streaming / tracing | Manual | `lc_agent_name` metadata, LangSmith-ready |
| When to use raw LangGraph | When the agent loop shape itself is custom | When you want the harness and focus on tools/prompts |
