"""
db_setup.py  —  SQLite seed data for orders_db and sales_db
Run once:  python db_setup.py
"""

import sqlite3, os

BASE = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(BASE, exist_ok=True)

ORDERS_DB = os.path.join(BASE, "orders.db")
SALES_DB  = os.path.join(BASE, "sales.db")


def init_orders_db():
    con = sqlite3.connect(ORDERS_DB)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id  INTEGER PRIMARY KEY,
        name         TEXT NOT NULL,
        email        TEXT,
        country      TEXT,
        signup_date  DATE
    );
    CREATE TABLE IF NOT EXISTS orders (
        order_id     INTEGER PRIMARY KEY,
        customer_id  INTEGER REFERENCES customers(customer_id),
        order_date   DATE,
        status       TEXT CHECK(status IN ('pending','confirmed','shipped','delivered','cancelled')),
        total_amount REAL
    );
    CREATE TABLE IF NOT EXISTS order_items (
        item_id      INTEGER PRIMARY KEY,
        order_id     INTEGER REFERENCES orders(order_id),
        product_id   INTEGER,
        product_name TEXT,
        quantity     INTEGER,
        unit_price   REAL
    );
    """)
    cur.executemany("INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?)", [
        (1, "Alice Johnson", "alice@example.com", "USA",   "2022-01-15"),
        (2, "Bob Smith",     "bob@example.com",   "UK",    "2022-03-20"),
        (3, "Carlos Diaz",   "carlos@example.com","Spain", "2022-06-01"),
        (4, "Diana Wei",     "diana@example.com", "China", "2023-01-10"),
        (5, "Evan Brown",    "evan@example.com",  "USA",   "2023-04-22"),
    ])
    cur.executemany("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?)", [
        (1001, 1, "2024-01-10", "delivered",  1500.00),
        (1002, 2, "2024-01-15", "delivered",  3200.00),
        (1003, 3, "2024-02-01", "shipped",     800.00),
        (1004, 4, "2024-02-20", "confirmed",  4500.00),
        (1005, 1, "2024-03-05", "delivered",  2100.00),
        (1006, 5, "2024-03-10", "pending",     950.00),
        (1007, 2, "2024-03-15", "cancelled",  1200.00),
        (1008, 3, "2024-04-01", "delivered",   670.00),
    ])
    cur.executemany("INSERT OR IGNORE INTO order_items VALUES (?,?,?,?,?,?)", [
        (1, 1001, 101, "Laptop Pro",      1, 1200.00),
        (2, 1001, 102, "Wireless Mouse",  2,  150.00),
        (3, 1002, 103, "4K Monitor",      2, 1600.00),
        (4, 1003, 101, "Laptop Pro",      1,  800.00),
        (5, 1004, 104, "Server Node",     1, 4500.00),
        (6, 1005, 105, "SSD 2TB",         3,  700.00),
        (7, 1006, 102, "Wireless Mouse",  5,  750.00),
        (8, 1007, 103, "4K Monitor",      1, 1200.00),
        (9, 1008, 106, "USB-C Hub",       2,  670.00),
    ])
    con.commit(); con.close()
    print(f"[orders_db] ready → {ORDERS_DB}")


def init_sales_db():
    con = sqlite3.connect(SALES_DB)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS products (
        product_id   INTEGER PRIMARY KEY,
        product_name TEXT NOT NULL,
        category     TEXT,
        unit_price   REAL,
        cost_price   REAL
    );
    CREATE TABLE IF NOT EXISTS sales_reps (
        rep_id  INTEGER PRIMARY KEY,
        name    TEXT NOT NULL,
        region  TEXT,
        quota   REAL
    );
    CREATE TABLE IF NOT EXISTS sales_transactions (
        txn_id     INTEGER PRIMARY KEY,
        rep_id     INTEGER REFERENCES sales_reps(rep_id),
        product_id INTEGER REFERENCES products(product_id),
        txn_date   DATE,
        quantity   INTEGER,
        revenue    REAL,
        region     TEXT
    );
    CREATE TABLE IF NOT EXISTS sales_targets (
        target_id     INTEGER PRIMARY KEY,
        region        TEXT,
        year          INTEGER,
        quarter       INTEGER,
        target_amount REAL
    );
    """)
    cur.executemany("INSERT OR IGNORE INTO products VALUES (?,?,?,?,?)", [
        (101, "Laptop Pro",     "Computers",      1200.00, 700.00),
        (102, "Wireless Mouse", "Peripherals",      75.00,  20.00),
        (103, "4K Monitor",     "Displays",        800.00, 400.00),
        (104, "Server Node",    "Infrastructure", 4500.00,2200.00),
        (105, "SSD 2TB",        "Storage",         250.00, 100.00),
        (106, "USB-C Hub",      "Peripherals",      85.00,  25.00),
    ])
    cur.executemany("INSERT OR IGNORE INTO sales_reps VALUES (?,?,?,?)", [
        (1, "Sarah Connor", "North America", 500000),
        (2, "John Miles",   "Europe",        400000),
        (3, "Amy Zhang",    "Asia Pacific",  350000),
    ])
    cur.executemany("INSERT OR IGNORE INTO sales_transactions VALUES (?,?,?,?,?,?,?)", [
        (1, 1, 101, "2024-01-10",  5,  6000.00, "North America"),
        (2, 1, 103, "2024-01-15",  3,  2400.00, "North America"),
        (3, 2, 104, "2024-02-01",  1,  4500.00, "Europe"),
        (4, 3, 101, "2024-02-20",  8,  9600.00, "Asia Pacific"),
        (5, 2, 102, "2024-03-05", 20,  1500.00, "Europe"),
        (6, 1, 105, "2024-03-10", 12,  3000.00, "North America"),
        (7, 3, 106, "2024-04-01", 30,  2550.00, "Asia Pacific"),
        (8, 2, 103, "2024-04-10",  6,  4800.00, "Europe"),
    ])
    cur.executemany("INSERT OR IGNORE INTO sales_targets VALUES (?,?,?,?,?)", [
        (1, "North America", 2024, 1, 150000),
        (2, "North America", 2024, 2, 175000),
        (3, "Europe",        2024, 1, 120000),
        (4, "Europe",        2024, 2, 140000),
        (5, "Asia Pacific",  2024, 1, 100000),
        (6, "Asia Pacific",  2024, 2, 120000),
    ])
    con.commit(); con.close()
    print(f"[sales_db] ready → {SALES_DB}")


if __name__ == "__main__":
    init_orders_db()
    init_sales_db()
    print("All databases ready.")
