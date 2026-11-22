import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else "clinic.db"
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

print("Top 5 doctors by completed appointments")
for r in con.execute("""
    SELECT d.first_name, d.last_name, COUNT(*) AS c
    FROM appointments a JOIN doctors d ON d.id = a.doctor_id
    WHERE a.status = 'completed'
    GROUP BY d.id ORDER BY c DESC, d.last_name, d.first_name LIMIT 5"""):
    print(r)

print("2024 monthly invoiced")
for r in con.execute("""
    SELECT substr(issued_at, 1, 7), SUM(amount) FROM invoices
    WHERE issued_at >= '2024-01-01' AND issued_at < '2025-01-01'
    GROUP BY 1 ORDER BY 1"""):
    print(r)