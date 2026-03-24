import sqlite3

def update_db():
    conn = sqlite3.connect('agrisense.db')
    cursor = conn.cursor()
    
    try:
        # Check if mobile column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'mobile' not in columns:
            print("Adding 'mobile' column to 'users' table...")
            cursor.execute("ALTER TABLE users ADD COLUMN mobile TEXT")
            conn.commit()
            print("Column added successfully.")
        else:
            print("'mobile' column already exists.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    update_db()
