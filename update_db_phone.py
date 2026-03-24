import sqlite3

def update_db():
    conn = sqlite3.connect('agrisense.db')
    cursor = conn.cursor()
    
    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'phone_number' not in columns:
            print("Adding phone_number column to users table...")
            cursor.execute("ALTER TABLE users ADD COLUMN phone_number TEXT")
            print("Column added successfully.")
        else:
            print("phone_number column already exists.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.commit()
        conn.close()

if __name__ == "__main__":
    update_db()
