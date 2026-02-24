import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from app.config import DB_CONFIG

# Create pool at import time (app startup)
connection_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    **DB_CONFIG
)

def execute_query(query: str):
    conn = connection_pool.getconn()   # Borrow connection
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            return results
    
    finally:
        connection_pool.putconn(conn)  # Return connection to pool