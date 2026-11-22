import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else "clinic.db"
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

print("Doctors named Smith:")
for r in con.execute("SELECT id, first_name, last_name, specialty FROM doctors WHERE last_name = 'Smith'"):
    print(r)

print("March 2024 appointments per Smith:")
q = """SELECT d.id, d.first_name, COUNT(*) FROM appointments a
       JOIN doctors d ON d.id = a.doctor_id
       WHERE d.last_name = 'Smith'
         AND a.scheduled_at >= '2024-03-01' AND a.scheduled_at < '2024-04-01'
       GROUP BY d.id"""
for r in con.execute(q):
    print(r)