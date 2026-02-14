from sqlalchemy import create_engine, text, inspect
import os
from dotenv import load_dotenv

load_dotenv(override=True)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "root")
DB_NAME = os.getenv("DB_NAME", "story_db")

# Use MySQL connection string
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)

def run_migration():
    inspector = inspect(engine)
    columns = [c['name'] for c in inspector.get_columns('stories')]
    
    with engine.connect() as conn:
        # 1. Add summary column to stories if missing
        if 'summary' not in columns:
            print("Adding 'summary' column to 'stories' table...")
            conn.execute(text("ALTER TABLE stories ADD COLUMN summary LONGTEXT NULL"))
            conn.commit()
            print("Added 'summary' column.")
        else:
            print("'summary' column already exists.")
            
        # 2. Add description column to stories if missing (was in model but maybe not in DB)
        if 'description' not in columns:
            print("Adding 'description' column to 'stories' table...")
            conn.execute(text("ALTER TABLE stories ADD COLUMN description TEXT NULL"))
            conn.commit()
            print("Added 'description' column.")

        # 3. Ensure other tables exist (create_all usually handles this, but let's be safe)
        from app.db.connection import Base
        from app.db import models
        Base.metadata.create_all(bind=engine)
        print("Ensured all tables are created.")

if __name__ == "__main__":
    try:
        run_migration()
        print("Migration successful.")
    except Exception as e:
        print(f"Migration failed: {e}")
