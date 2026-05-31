"""
debug_stream.py
---------------
Run this INSTEAD of main.py to see every raw chunk deepagents emits.
This tells us exactly what keys/types carry the planning data.

Usage:
    python debug_stream.py
"""

from dotenv import load_dotenv
load_dotenv()

import json, pprint
from langchain_core.messages import HumanMessage
from db_setup import init_orders_db, init_sales_db
from orchestrator import build_orchestrator

init_orders_db()
init_sales_db()

orchestrator = build_orchestrator()
QUESTION = "Which customers have placed more than one order?"

print(f"\n{'='*70}")
print(f"DEBUG STREAM DUMP")
print(f"Q: {QUESTION}")
print(f"{'='*70}\n")

chunk_index = 0

# ── Try 1: messages + updates combined ───────────────────────────────────────
print("\n>>> MODE: stream_mode=['messages','updates'], subgraphs=True\n")

for chunk in orchestrator.stream(
    {"messages": [HumanMessage(content=QUESTION)]},
    stream_mode=["messages", "updates"],
    subgraphs=True,
    version="v2",
):
    chunk_index += 1
    chunk_type = chunk.get("type", "unknown") if isinstance(chunk, dict) else type(chunk).__name__
    ns         = chunk.get("ns", ()) if isinstance(chunk, dict) else ()

    # ── Print a compact summary of every chunk ──────────────
    print(f"\n--- chunk #{chunk_index}  type={chunk_type}  ns={ns} ---")

    if not isinstance(chunk, dict):
        print(f"  (raw) {repr(chunk)[:200]}")
        continue

    data = chunk.get("data")

    if chunk_type == "messages":
        token, meta = data if isinstance(data, tuple) else (data, {})
        msg_type     = getattr(token, "type", type(token).__name__)
        content      = getattr(token, "content", "") or ""
        tool_calls   = getattr(token, "tool_calls", []) or []
        tc_chunks    = getattr(token, "tool_call_chunks", []) or []
        lc_name      = (meta.get("metadata", {}).get("lc_agent_name")
                        or meta.get("lc_agent_name", ""))

        print(f"  msg_type    : {msg_type}")
        print(f"  lc_agent_name: {lc_name!r}")
        if content:
            print(f"  content     : {content[:120]!r}")
        if tool_calls:
            for tc in tool_calls:
                print(f"  tool_call   : name={tc.get('name')!r}  "
                      f"args_keys={list(tc.get('args',{}).keys())}")
                # Show full args for write_todos / task
                if tc.get("name") in ("write_todos", "task"):
                    print(f"    full args : {json.dumps(tc.get('args',{}), indent=4)[:600]}")
        if tc_chunks:
            for tc in tc_chunks:
                if tc.get("name"):
                    print(f"  tc_chunk    : name={tc.get('name')!r}  "
                          f"args={tc.get('args','')[:80]!r}")

    elif chunk_type == "updates":
        if isinstance(data, dict):
            for node_name, node_data in data.items():
                print(f"  node        : {node_name!r}")
                if isinstance(node_data, dict):
                    for k, v in node_data.items():
                        if k == "messages":
                            for m in (v or []):
                                mname   = getattr(m, "name", "")
                                mcont   = getattr(m, "content", "") or ""
                                mtype   = getattr(m, "type", "")
                                mtc     = getattr(m, "tool_calls", []) or []
                                print(f"    msg  type={mtype!r}  "
                                      f"name={mname!r}  "
                                      f"content={mcont[:100]!r}")
                                if mtc:
                                    for tc in mtc:
                                        print(f"      tc: {tc.get('name')!r} "
                                              f"args={json.dumps(tc.get('args',{}))[:200]}")
                        elif k == "todos":
                            print(f"    todos: {v!r}")
                        elif k == "lc_agent_name":
                            print(f"    lc_agent_name: {v!r}")
                        else:
                            val_repr = repr(v)[:120]
                            print(f"    {k}: {val_repr}")
        else:
            print(f"  data: {repr(data)[:300]}")

    else:
        print(f"  raw: {repr(chunk)[:300]}")

print(f"\n{'='*70}")
print(f"Total chunks: {chunk_index}")
print(f"{'='*70}\n")