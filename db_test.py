import psycopg
try:
    conn = psycopg.connect("host=localhost port=5433 user=postgres password=mypassword")
    print("✅ SUCCESS: The door is open!")
    conn.close()
except Exception as e:
    print(f"❌ FAILED: {e}")