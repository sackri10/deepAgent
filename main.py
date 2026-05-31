"""
main.py  —  Run the full multi-agent Text-to-SQL system.

Usage:
    OPENAI_API_KEY=sk-...  python main.py

Or put the key in a .env file (loaded automatically).
"""

import os
from dotenv import load_dotenv
load_dotenv()

from db_setup import init_orders_db, init_sales_db
from streaming import ask_streaming


def main():
    print("── Initialising databases ──────────────────────────────────")
    init_orders_db()
    init_sales_db()

    questions = [
        # Cross-domain: both agents fire
        "What are the top 3 products by revenue in sales, "
        "and how many customer orders included each of those products?",

        # Orders only
        "Which customers have placed more than one order and what is their total spend?",

        # Sales only
        "Which sales rep exceeded their Q1 2024 quota and by how much?",
    ]

    for q in questions:
        ask_streaming(q)
        input("\n  ↵  Press Enter for next question...")


    # # ── Cross-domain: both agents fire
    # run(
    #     "Cross-domain",
    #     "What are the top 3 products by revenue in sales, "
    #     "and how many customer orders included each of those products?"
    # )

    # # ── Orders only
    # run(
    #     "Orders only",
    #     "Which customers have placed more than one order? "
    #     "Show their total spend and order count."
    # )

    # # ── Sales only
    # run(
    #     "Sales only",
    #     "Which sales rep has the highest revenue in Q1 2024, "
    #     "and by how much did they exceed or miss their quarterly target?"
    # )

    # # ── Aggregation across domains
    # run(
    #     "Cross-domain aggregation",
    #     "Compare the revenue from sales transactions with the total order value "
    #     "from confirmed and delivered orders. Which is higher and by how much?"
    # )


if __name__ == "__main__":
    main()
