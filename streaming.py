"""
streaming.py
------------
Clean, readable streaming of the full multi-agent thinking process.

Output sections (in order):
  ┌─────────────────────────────────────────────────┐
  │  QUESTION                                       │
  ├─────────────────────────────────────────────────┤
  │  [Orchestrator] thinking tokens                 │
  │  [Orchestrator] → delegating to orders-agent    │
  │  [Orchestrator] → delegating to sales-agent     │
  ├─────────────────────────────────────────────────┤
  │  [orders-agent] planning / thinking tokens      │
  │  [orders-agent] SQL tool call + result          │
  │  [orders-agent] ✓ done                          │
  ├─────────────────────────────────────────────────┤
  │  [sales-agent]  planning / thinking tokens      │
  │  [sales-agent]  SQL tool call + result          │
  │  [sales-agent]  ✓ done                          │
  ├─────────────────────────────────────────────────┤
  │  [Orchestrator] final answer tokens             │
  └─────────────────────────────────────────────────┘

Uses stream_mode="messages" + subgraphs=True which works on all
deepagents versions and gives token-level granularity.
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage
 
# ── ANSI colours ─────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
RED     = "\033[31m"
 
AGENT_COLOURS = {
    "text-to-sql-orchestrator": CYAN,
    "orders-agent":             GREEN,
    "sales-agent":              YELLOW,
    "general-purpose":          BLUE,
}
 
def _c(agent: str) -> str:
    return AGENT_COLOURS.get(agent, MAGENTA)
 
def _tag(agent: str) -> str:
    return f"{_c(agent)}{BOLD}[{agent}]{RESET}"
 
 
# ── State carried across chunks ───────────────────────────────
 
class _State:
    def __init__(self):
        self.cur_agent   = ""    # last agent we printed a section for
        self.cur_section = ""    # "thinking" | "tool:<name>" | "answer" | ""
        self.got_subagent = False  # have any sub-agents spoken yet?
        self.pending_nl  = False   # need to print \n before next header
 
    def newline(self):
        if self.pending_nl:
            print()
            self.pending_nl = False
 
    def divider(self):
        self.newline()
        print(f"\n{DIM}{'─' * 64}{RESET}")
 
    def section(self, agent: str, kind: str, extra: str = ""):
        """
        Print a section header only when agent or section type changes.
        kind: "thinking" | "tool:<name>" | "answer" | "delegating"
        """
        if agent == self.cur_agent and kind == self.cur_section:
            return   # still in same section — keep streaming inline
 
        self.newline()
 
        # Divider when switching between agents
        if agent != self.cur_agent and self.cur_agent:
            self.divider()
 
        self.cur_agent   = agent
        self.cur_section = kind
        tag = _tag(agent)
        c   = _c(agent)
 
        if kind == "thinking":
            print(f"\n{tag} 💭 ", end="", flush=True)
        elif kind.startswith("tool:"):
            name = kind.split(":", 1)[1]
            print(f"\n{tag} 🔧  {BOLD}{name}{RESET}", end="", flush=True)
            if extra:
                print(f"\n    {DIM}{extra}{RESET}", end="", flush=True)
        elif kind == "delegating":
            print(f"\n{tag} 📤  delegating → {BOLD}{extra}{RESET}")
        elif kind == "answer":
            print(f"\n{tag} ✅  final answer:\n")
        elif kind == "result":
            print(f"\n{tag} 📋  ", end="", flush=True)
 
        self.pending_nl = False
 
 
# ── Main streaming function ───────────────────────────────────
 
def stream_ask(question: str, orchestrator) -> None:
    """
    Stream the full agent run with clean, labelled output.
    Uses lc_agent_name from chunk metadata — the official deepagents API.
    """
    st = _State()
 
    # ── Question header
    bar = "═" * 64
    print(f"\n{CYAN}{bar}{RESET}")
    print(f"{CYAN}  Q: {BOLD}{question}{RESET}")
    print(f"{CYAN}{bar}{RESET}")
 
    # Track: has any sub-agent spoken yet? Used to detect final-answer phase.
    seen_agents: set[str] = set()
 
    for chunk in orchestrator.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="messages",
        subgraphs=True,
        version="v2",
    ):
        if not isinstance(chunk, dict) or chunk.get("type") != "messages":
            continue
 
        token, metadata = chunk["data"]
 
        # ── Resolve agent name from metadata  ← THE KEY CHANGE
        # deepagents sets lc_agent_name on every message it emits
        agent: str = (
            metadata.get("metadata", {}).get("lc_agent_name")
            or metadata.get("lc_agent_name")
            or "text-to-sql-orchestrator"
        )
 
        msg_type         = getattr(token, "type", "")
        content: str     = getattr(token, "content", "") or ""
        tool_calls       = getattr(token, "tool_calls", []) or []
        tool_call_chunks = getattr(token, "tool_call_chunks", []) or []
 
        seen_agents.add(agent)
        is_orchestrator = agent == "text-to-sql-orchestrator"
        subagents_done  = len(seen_agents) > 1  # at least one sub-agent has spoken
 
        # ── AIMessageChunk ────────────────────────────────────
        if "AI" in msg_type:
 
            # ── Full tool_calls (non-streaming) ───────────────
            if tool_calls:
                for tc in tool_calls:
                    name    = tc.get("name", "")
                    args    = tc.get("args", {})
 
                    if name == "task":
                        # Orchestrator delegating to a sub-agent
                        sub_name = args.get("name", "?")
                        task_str = args.get("task", "")
                        st.section(agent, "delegating", sub_name)
                        print(f"    {DIM}task: {task_str}{RESET}", flush=True)
                    else:
                        # Domain agent calling a SQL tool
                        query = (
                            args.get("query", "")
                            or args.get("sql", "")
                            or str(args)
                        )
                        st.section(agent, f"tool:{name}", query[:200])
 
            # ── Streaming tool_call_chunks ────────────────────
            elif tool_call_chunks:
                for tc in tool_call_chunks:
                    name = tc.get("name") or ""
                    args = tc.get("args") or ""
 
                    if name:
                        if name == "task":
                            # Will get sub-agent name in args stream
                            st.section(agent, "delegating", "…")
                        else:
                            st.section(agent, f"tool:{name}")
 
                    if args and st.cur_section.startswith("tool:"):
                        # Streaming SQL query characters
                        print(f"{DIM}{args}{RESET}", end="", flush=True)
                        st.pending_nl = True
                    elif args and st.cur_section == "delegating":
                        # Streaming task args — extract sub-agent name if visible
                        import re
                        m = re.search(r'"name"\s*:\s*"([^"]+)"', args)
                        if m:
                            sub = m.group(1)
                            st.newline()
                            print(
                                f"\n{_tag(agent)} 📤  delegating → {BOLD}{sub}{RESET}",
                                flush=True
                            )
                            st.cur_section = "delegating_named"
 
            # ── Plain text (thinking / final answer) ──────────
            elif content:
                is_final = is_orchestrator and subagents_done
                kind = "answer" if is_final else "thinking"
                st.section(agent, kind)
                print(content, end="", flush=True)
                st.pending_nl = True
 
        # ── ToolMessage (result returned from tool) ───────────
        elif msg_type == "ToolMessage":
            tool_name = getattr(token, "name", "") or "tool"
            # Truncate large results (raw SQL rows can be huge)
            display = content.replace("\n", " ")
            if len(display) > 280:
                display = display[:280] + f"… [{len(content)} chars total]"
            st.section(agent, "result")
            print(f"{DIM}{display}{RESET}", flush=True)
 
    # ── Final newline + footer ────────────────────────────────
    st.newline()
    print(f"\n{GREEN}{bar}{RESET}\n")
 
 
# ── Convenience wrapper ───────────────────────────────────────
 
def ask_streaming(question: str) -> None:
    """
    Drop-in replacement for ask() that streams with clean output.
    Usage:
        from streaming import ask_streaming
        ask_streaming("Which region exceeded its Q1 sales target?")
    """
    from orchestrator import build_orchestrator
    stream_ask(question, build_orchestrator())