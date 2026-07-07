import sqlite3


DATABASE_FILE = "database.db"


conn = sqlite3.connect(DATABASE_FILE)
cursor = conn.cursor()

cursor.execute("""
    SELECT id, employee_name, date, time, check_type, confidence, image_path
    FROM attendance_logs
    ORDER BY id DESC
""")

rows = cursor.fetchall()

conn.close()

print("Danh sach diem danh trong database:")
print("--------------------------------")

for row in rows:
    print(row)