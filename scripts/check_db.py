import sqlite3

con = sqlite3.connect("file:clinic.db?mode=ro", uri=True)
checks = {
    "orphan appointments->patients": "SELECT COUNT(*) FROM appointments a LEFT JOIN patients p ON p.id=a.patient_id WHERE p.id IS NULL",
    "orphan appointments->doctors": "SELECT COUNT(*) FROM appointments a LEFT JOIN doctors d ON d.id=a.doctor_id WHERE d.id IS NULL",
    "orphan treatments->appointments": "SELECT COUNT(*) FROM treatments t LEFT JOIN appointments a ON a.id=t.appointment_id WHERE a.id IS NULL",
    "orphan payments->invoices": "SELECT COUNT(*) FROM payments p LEFT JOIN invoices i ON i.id=p.invoice_id WHERE i.id IS NULL",
    "invoices on non-completed appts (expect 0)": "SELECT COUNT(*) FROM invoices i JOIN appointments a ON a.id=i.appointment_id WHERE a.status!='completed'",
    "invoice amount != treatment sum (expect 0)": "SELECT COUNT(*) FROM invoices i WHERE ABS(i.amount-(SELECT SUM(cost) FROM treatments t WHERE t.appointment_id=i.appointment_id))>0.01",
    "payment > invoice amount (expect 0)": "SELECT COUNT(*) FROM payments p JOIN invoices i ON i.id=p.invoice_id WHERE p.amount>i.amount+0.01",
    "payment before invoice issued (expect 0)": "SELECT COUNT(*) FROM payments p JOIN invoices i ON i.id=p.invoice_id WHERE p.payment_date<i.issued_at",
    "strftime returns NULL (expect 0)": "SELECT COUNT(*) FROM appointments WHERE strftime('%Y-%m', scheduled_at) IS NULL",
}
for name, sql in checks.items():
    print(f"{name}: {con.execute(sql).fetchone()[0]}")
print("status %:", con.execute("SELECT status, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM appointments),1) FROM appointments GROUP BY status").fetchall())
print("date range:", con.execute("SELECT MIN(scheduled_at), MAX(scheduled_at) FROM appointments").fetchone())
print("patients cols:", [r[1] for r in con.execute("PRAGMA table_info(patients)")])

print("Q1 completed:", con.execute("SELECT COUNT(*) FROM appointments WHERE status='completed'").fetchone())
print("Q3 completed 2024:", con.execute("SELECT COUNT(*) FROM appointments WHERE status='completed' AND strftime('%Y',scheduled_at)='2024'").fetchone())
print("Q2 invoiced by specialty:", con.execute("SELECT d.specialty, ROUND(SUM(i.amount),2) FROM invoices i JOIN appointments a ON a.id=i.appointment_id JOIN doctors d ON d.id=a.doctor_id GROUP BY d.specialty ORDER BY 2 DESC").fetchall())
