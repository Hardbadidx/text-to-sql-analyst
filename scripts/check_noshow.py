import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else "clinic.db"
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
cols = [r[1] for r in con.execute("PRAGMA table_info(appointments)")]
print("columns:", cols)
date_col = next(c for c in cols if "date" in c or c.endswith("_at") or "time" in c)
print("using:", date_col)
q = (f"SELECT substr({date_col},1,4), COUNT(*), SUM(status='no_show') "
     "FROM appointments GROUP BY 1")
for year, total, ns in con.execute(q):
    print(year, total, ns, round(100 * ns / total, 2))
    