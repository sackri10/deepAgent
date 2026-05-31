"""
streaming.py
------------
Clean streaming based on actual observed chunk structure from debug_stream.py.

Key facts discovered from the debug dump:
  1. task() args use keys: "subagent_type" and "description" (NOT "name"/"task")
  2. Anthropic content is a LIST of dicts: [{"text":"..","type":"text"}]
     or [{"partial_json":"..","type":"input_json_delta"}]
  3. write_todos is NOT called by this model/prompt — no planning tool fires
  4. TodoListMiddleware.after_model fires but carries no data — skip it
  5. lc_agent_name works correctly on every AIMessageChunk
  6. Sub-agent namespace: ('tools:<uuid>',) — resolve via lc_agent_name directly
  7. Full tool args arrive in "updates" node='model' chunk — use that for
     complete SQL query display (not fragmented partial_json)
  8. Sub-agent final answer arrives as ToolMessage at orchestrator level (chunk #107)

What we display:
  [orchestrator]  🚀  delegating → orders-agent   ← from task() tool_call
                      task: "Which customers..."   ← from description arg
  ── orders-agent ─────────────────────────────
  [orders-agent]  💬  I'll help find customers...  ← text content
  [orders-agent]  🔧  orders_get_schema            ← tool call starts
  [orders-agent]  📋  result: Database: orders.db  ← ToolMessage
  [orders-agent]  🔧  orders_execute_sql
                      SELECT c.customer_id...       ← full SQL from updates
  [orders-agent]  📋  result: Alice Johnson (2)...  ← ToolMessage
  [orders-agent]  💬  **Customers with Multiple... ← final answer text
  ── back to orchestrator ─────────────────────
  [orchestrator]  ✅  Three customers have placed...← final synthesis
"""

from __future__ import annotations
import json
from langchain_core.messages import HumanMessage

# ── ANSI colours ──────────────────────────────────────────────
R   = "\033[0m"      # reset
B   = "\033[1m"      # bold
D   = "\033[2m"      # dim
CY  = "\033[36m"     # cyan    — orchestrator
GR  = "\033[32m"     # green   — orders-agent
YL  = "\033[33m"     # yellow  — sales-agent
BL  = "\033[34m"     # blue    — general-purpose
MG  = "\033[35m"     # magenta — unknown

COLOURS = {
    "text-to-sql-orchestrator": CY,
    "orders-agent":             GR,
    "sales-agent":              YL,
    "general-purpose":          BL,
}

def _c(agent: str) -> str:
    return COLOURS.get(agent, MG)

def _tag(agent: str) -> str:
    return f"{_c(agent)}{B}[{agent}]{R}"


# ── Extract readable text from Anthropic content blocks ───────

def _text_from_content(content) -> str:
    """
    Anthropic returns content as a list of typed dicts:
      [{"text": "hello", "type": "text"}]
      [{"partial_json": '{"query":', "type": "input_json_delta"}]
    Plain strings are passed through as-is.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "input_json_delta":
                    parts.append(block.get("partial_json", ""))
        return "".join(parts)
    return ""


def _sql_from_args(args) -> str:
    """Extract SQL from tool call args dict."""
    if isinstance(args, dict):
        return args.get("query", args.get("sql", ""))
    return ""


# ── Stream state ──────────────────────────────────────────────

class _S:
    """Mutable streaming state passed across chunks."""
    def __init__(self):
        self.cur_agent   = ""
        self.cur_section = ""   # "text" | "tool:<name>" | "result" | "answer"
        self.nl_pending  = False
        self.seen_agents: set[str] = set()
        # Cache full SQL from updates chunks to display cleanly
        self.pending_sql: dict[str, str] = {}   # tool_call_id → full SQL

    def flush(self):
        if self.nl_pending:
            print()
            self.nl_pending = False

    def agent_banner(self, agent: str):
        """Print a divider banner when switching to a new sub-agent."""
        if agent == self.cur_agent:
            return
        self.flush()
        c = _c(agent)
        if self.cur_agent:   # not the very first agent
            print(f"\n{D}{'─' * 64}{R}")
        if agent != "text-to-sql-orchestrator":
            print(f"{c}{B}{'─' * 24} {agent} {'─' * (38 - len(agent))}{R}")
        self.cur_agent   = agent
        self.cur_section = ""

    def header(self, agent: str, kind: str, detail: str = ""):
        """
        Print section header only on transitions.
        kind: "text" | "tool:<name>" | "result" | "delegating" | "answer"
        """
        if agent == self.cur_agent and kind == self.cur_section:
            return
        self.flush()
        self.agent_banner(agent)
        self.cur_section = kind
        tag = _tag(agent)

        if kind == "text":
            print(f"\n{tag} 💬  ", end="", flush=True)

        elif kind == "answer":
            print(f"\n{tag} ✅  ", end="", flush=True)

        elif kind.startswith("tool:"):
            name = kind.split(":", 1)[1]
            print(f"\n{tag} 🔧  {B}{name}{R}", end="", flush=True)
            if detail:
                print(f"\n    {D}{detail}{R}", end="", flush=True)

        elif kind == "result":
            print(f"\n{tag} 📋  {D}", end="", flush=True)

        elif kind == "delegating":
            print(f"\n{tag} 🚀  delegating → {B}{detail}{R}")

        self.nl_pending = False


# ── Main function ─────────────────────────────────────────────

def stream_ask(question: str, orchestrator) -> None:
    bar = "═" * 64
    print(f"\n{CY}{bar}{R}")
    print(f"{CY}  {B}Q: {question}{R}")
    print(f"{CY}{bar}{R}\n")

    s = _S()

    # We need the full SQL from updates (not fragmented partial_json).
    # Cache: tool_call_id → full query string, populated from updates chunks.
    sql_cache: dict[str, str] = {}
    todos_cache: dict[str, list] = {}   # tool_call_id → full todos list

    # Track which tool_call_id → tool_name (to know when SQL tool fires)
    tool_id_to_name: dict[str, str] = {}

    for chunk in orchestrator.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode=["messages", "updates"],
        subgraphs=True,
        version="v2",
    ):
        if not isinstance(chunk, dict):
            continue

        ctype = chunk.get("type")
        ns    = chunk.get("ns", ())

        # ── Resolve agent name ────────────────────────────────
        agent = "text-to-sql-orchestrator"
        if ctype == "messages":
            _, meta = chunk.get("data", (None, {}))
            agent = (
                meta.get("metadata", {}).get("lc_agent_name")
                or meta.get("lc_agent_name")
                or "text-to-sql-orchestrator"
            )

        s.seen_agents.add(agent)
        is_orch   = (agent == "text-to-sql-orchestrator")
        has_sub   = len(s.seen_agents) > 1

        # ══════════════════════════════════════════════════════
        # UPDATES — extract full SQL from completed tool calls
        # ══════════════════════════════════════════════════════
        if ctype == "updates":
            data = chunk.get("data", {})
            if not isinstance(data, dict):
                continue

            for node_name, node_data in data.items():
                # Skip empty middleware nodes
                if not isinstance(node_data, dict) or not node_data:
                    continue

                # Extract full tool call args from completed model node
                msgs = node_data.get("messages", [])
                for msg in msgs:
                    tcs = getattr(msg, "tool_calls", []) or []
                    for tc in tcs:
                        tc_id   = tc.get("id", "")
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("args", {})
                        if tc_id:
                            tool_id_to_name[tc_id] = tc_name
                        sql = _sql_from_args(tc_args)
                        if sql and tc_id:
                            sql_cache[tc_id] = sql
                        # Cache todos args for write_todos tool calls
                        if tc_name == "write_todos" and tc_id:
                            todos_val = (
                                tc_args.get("todos")
                                or tc_args.get("tasks")
                                or tc_args.get("items")
                            )
                            if todos_val:
                                todos_cache[tc_id] = todos_val
            continue   # updates handled, move on

        # ══════════════════════════════════════════════════════
        # MESSAGES — render token-by-token output
        # ══════════════════════════════════════════════════════
        if ctype != "messages":
            continue

        token, _ = chunk.get("data", (None, None))
        if token is None:
            continue

        msg_type    = getattr(token, "type", "")
        raw_content = getattr(token, "content", "") or ""
        tool_calls  = getattr(token, "tool_calls", []) or []
        tc_chunks   = getattr(token, "tool_call_chunks", []) or []

        # ── AIMessageChunk ────────────────────────────────────
        if "AI" in msg_type:

            # ── Full tool_calls (non-streaming, arrives at once)
            if tool_calls:
                for tc in tool_calls:
                    name    = tc.get("name", "")
                    args    = tc.get("args", {})
                    tc_id   = tc.get("id", "")

                    if name == "write_todos":
                        pass  # PLAN rendered from ToolMessage (has full data)

                    elif name == "task":
                        # Delegation: args have "subagent_type" + "description"
                        sub   = args.get("subagent_type", args.get("name", "?"))
                        desc  = args.get("description",   args.get("task", ""))
                        s.header(agent, "delegating", sub)
                        if desc:
                            print(f"    {D}task: {desc}{R}", flush=True)

                    else:
                        # SQL or schema tool — show full SQL from cache
                        sql = sql_cache.get(tc_id, "") or _sql_from_args(args)
                        s.header(agent, f"tool:{name}", sql[:400] if sql else "")

            # ── Streaming tool_call_chunks (name arrives first, then args)
            elif tc_chunks:
                for tc in tc_chunks:
                    name      = tc.get("name") or ""
                    args_frag = _text_from_content(raw_content)

                    if name:
                        if name == "write_todos":
                            pass  # PLAN rendered from ToolMessage
                        elif name == "task":
                            s.header(agent, "delegating", "…")
                        else:
                            s.header(agent, f"tool:{name}")

                    # Stream partial_json for task description live
                    if args_frag and s.cur_section == "delegating":
                        # Try to fish out subagent_type and description
                        import re
                        m = re.search(
                            r'"subagent_type"\s*:\s*"([^"]+)"', args_frag
                        )
                        if m:
                            sub = m.group(1)
                            s.flush()
                            print(
                                f"\n{_tag(agent)} 🚀  "
                                f"delegating → {B}{sub}{R}",
                                flush=True
                            )
                            s.cur_section = "delegating_named"

            # ── Plain text content (thinking / final answer)
            else:
                text = _text_from_content(raw_content)
                if text:
                    is_final = is_orch and has_sub
                    kind = "answer" if is_final else "text"
                    s.header(agent, kind)
                    print(text, end="", flush=True)
                    s.nl_pending = True

        # ── ToolMessage — result returned from a tool ─────────
        elif msg_type == "tool":
            tool_name = getattr(token, "name", "") or "tool"

            # write_todos ToolMessage has the FULL rendered todo list — render PLAN here
            if tool_name == "write_todos":
                text = _text_from_content(raw_content) or str(raw_content)
                # deepagents returns "Updated todo list to [...]" — parse the list
                import re, ast
                m = re.search(r'Updated todo list to (\[.+)', text, re.DOTALL)
                todos_raw = None
                if m:
                    try:
                        todos_raw = ast.literal_eval(m.group(1).strip())
                    except Exception:
                        todos_raw = None
                s.flush()
                tag = _tag(agent)
                c   = _c(agent)
                print(f"\n{tag} 📋  {B}PLAN{R}")
                if isinstance(todos_raw, list):
                    for i, item in enumerate(todos_raw, 1):
                        if isinstance(item, dict):
                            status = item.get("status", "pending")
                            label  = (item.get("content")
                                      or item.get("task")
                                      or item.get("description")
                                      or str(item))
                            done   = status in ("completed", "done")
                            tick   = f"{GR}✓{R}" if done else (
                                     f"{YL}…{R}" if status == "in_progress" else "□")
                            dim    = D if done else ""
                            print(f"  {c}│{R}  {tick} {dim}{i}. {label}{R}")
                        else:
                            print(f"  {c}│{R}  □ {i}. {item}")
                else:
                    print(f"  {c}│{R}  {D}{text[:200]}{R}")
                s.cur_section = "plan"
                continue

            text      = _text_from_content(raw_content) or raw_content

            # Compact display: trim whitespace, cap length
            display = " ".join(text.split())
            if len(display) > 320:
                display = display[:320] + f"… [{len(text)} chars]"

            s.flush()
            s.header(agent, "result")
            print(f"{tool_name} → {display}{R}", flush=True)
            s.cur_section = ""   # reset so next tool header always prints

    s.flush()
    print(f"\n{CY}{bar}{R}\n")


# ── Convenience wrapper ───────────────────────────────────────

def ask_streaming(question: str) -> None:
    """
    Usage:
        from streaming import ask_streaming
        ask_streaming("Which region exceeded its Q1 2024 sales target?")
    """
    from orchestrator import build_orchestrator
    stream_ask(question, build_orchestrator())