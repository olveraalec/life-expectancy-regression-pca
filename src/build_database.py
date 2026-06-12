"""
build_database.py

Purpose:
Build the DuckDB database for the life expectancy project.

This script:
1. Connects to the local DuckDB database.
2. Runs the SQL file that loads the raw CSV.
3. Runs the SQL file that creates the clean SQL view.
4. Runs the SQL file that creates the modeling SQL view.
5. Confirms that the modeling view was created successfully.
"""

# Import DuckDB
import duckdb

from pathlib import Path

# Define the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the database file path
DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "life_expectancy.duckdb"

# Define the SQL folder path
SQL_DIR = PROJECT_ROOT / "sql"

# Define the SQL files that must be run to build the database
SQL_FILES = [
    SQL_DIR / "01_create_raw_table.sql",
    SQL_DIR / "02_create_clean_view.sql",
    SQL_DIR / "04_create_modeling_view.sql",
] 

def run_sql_file(connection, sql_file_path):
    """
    Run one SQL file using an active DuckDB connection.

    Parameters
    ----------
    connection:
        Active DuckDB connection.

    sql_file_path:
        Path to the SQL file we want to run.
    """

    # Read the SQL file as plain text.
    sql_text = sql_file_path.read_text()

    # Execute the SQL text using DuckDB.
    connection.execute(sql_text)

    # Print a progress message.
    print(f"Executed: {sql_file_path.name}")


def build_database():
    """
    Build the DuckDB database by running the SQL setup files.
    """

    # Make sure the database folder exists before connecting
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Connect to the DuckDB database
    conn = duckdb.connect(DATABASE_PATH)

    try:
        # Run each SQL file in the correct order
        for sql_file in SQL_FILES:
            run_sql_file(conn, sql_file)

        # Query the modeling view to confirm it exists and has rows
        row_count = conn.execute("""
            SELECT COUNT(*) AS num_rows
            FROM modeling_life_expectancy;
        """).fetchone()[0]

        # Print a success message
        print(f"Database built successfully.")
        print(f"Modeling view row count: {row_count}")

    finally:
        # Always close the database connection when finished
        conn.close()


# This block runs only when the file is executed directly
if __name__ == "__main__":
    build_database()