"""
export_dashboard_data.py

Purpose:
Export dashboard-ready CSV files from the DuckDB database

This script:
1. Connects to the local DuckDB database
2. Queries the clean_life_expectancy SQL view
3. Creates summary tables for Power BI
4. Saves those tables as CSV files in data/processed/
"""

# Import DuckDB
import duckdb

# Import Path
from pathlib import Path


# Define the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the path to the DuckDB database
DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "life_expectancy.duckdb"

# Define the folder where dashboard-ready CSV files will be saved
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def export_query_to_csv(connection, query, output_path):
    """
    Run a SQL query and export the result to a CSV file

    Parameters
    ----------
    connection:
        Active DuckDB connection

    query:
        SQL query string to execute

    output_path:
        File path where the CSV should be saved
    """

    # Execute the SQL query and return the result as a pandas DataFrame
    df = connection.execute(query).fetchdf()

    # Save the DataFrame as a CSV file
    df.to_csv(output_path, index=False)

    # Print a confirmation message
    print(f"Saved: {output_path.name} | Rows: {len(df)}")


def export_dashboard_data():
    """
    Export all dashboard-ready CSV files for Power BI
    """

    # Make sure the processed data folder exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to the existing DuckDB database
    conn = duckdb.connect(DATABASE_PATH)

    try:
        # Query 1:
        # Dataset overview summary
        overview_query = """
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT country) AS unique_countries,
            MIN(year) AS first_year,
            MAX(year) AS last_year,
            COUNT(DISTINCT status) AS unique_status_values,
            ROUND(AVG(life_expectancy), 2) AS avg_life_expectancy
        FROM clean_life_expectancy;
        """

        export_query_to_csv(
            connection=conn,
            query=overview_query,
            output_path=PROCESSED_DIR / "overview_summary.csv"
        )

        # Query 2:
        # Status-level summary
        # Useful for comparing Developed vs Developing countries in Power BI
        status_query = """
        SELECT
            status,
            COUNT(*) AS num_records,
            ROUND(AVG(life_expectancy), 2) AS avg_life_expectancy,
            ROUND(MIN(life_expectancy), 2) AS min_life_expectancy,
            ROUND(MAX(life_expectancy), 2) AS max_life_expectancy,
            ROUND(STDDEV(life_expectancy), 2) AS std_life_expectancy
        FROM clean_life_expectancy
        GROUP BY status
        ORDER BY avg_life_expectancy DESC;
        """

        export_query_to_csv(
            connection=conn,
            query=status_query,
            output_path=PROCESSED_DIR / "status_summary.csv"
        )

        # Query 3:
        # Year-level summary
        year_query = """
        SELECT
            year,
            COUNT(*) AS num_records,
            ROUND(AVG(life_expectancy), 2) AS avg_life_expectancy,
            ROUND(MIN(life_expectancy), 2) AS min_life_expectancy,
            ROUND(MAX(life_expectancy), 2) AS max_life_expectancy
        FROM clean_life_expectancy
        GROUP BY year
        ORDER BY year;
        """

        export_query_to_csv(
            connection=conn,
            query=year_query,
            output_path=PROCESSED_DIR / "year_summary.csv"
        )

        # Query 4:
        # Country-level summary
        # Useful for country ranking visuals in Power BI
        country_query = """
        SELECT
            country,
            status,
            COUNT(*) AS num_records,
            MIN(year) AS first_year,
            MAX(year) AS last_year,
            ROUND(AVG(life_expectancy), 2) AS avg_life_expectancy,
            ROUND(MIN(life_expectancy), 2) AS min_life_expectancy,
            ROUND(MAX(life_expectancy), 2) AS max_life_expectancy,
            ROUND(AVG(gdp), 2) AS avg_gdp,
            ROUND(AVG(schooling), 2) AS avg_schooling,
            ROUND(AVG(adult_mortality), 2) AS avg_adult_mortality
        FROM clean_life_expectancy
        GROUP BY country, status
        ORDER BY avg_life_expectancy DESC;
        """

        export_query_to_csv(
            connection=conn,
            query=country_query,
            output_path=PROCESSED_DIR / "country_summary.csv"
        )

        # Query 5:
        # Missingness summary
        # Useful for documenting data quality and showing why imputation was needed
        missingness_query = """
        WITH total AS (
            SELECT COUNT(*) AS total_rows
            FROM clean_life_expectancy
        ),

        missing_counts AS (
            SELECT 'adult_mortality' AS column_name, SUM(CASE WHEN adult_mortality IS NULL THEN 1 ELSE 0 END) AS missing_count FROM clean_life_expectancy
            UNION ALL
            SELECT 'alcohol', SUM(CASE WHEN alcohol IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
            UNION ALL
            SELECT 'percentage_expenditure', SUM(CASE WHEN percentage_expenditure IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
            UNION ALL
            SELECT 'hepatitis_b', SUM(CASE WHEN hepatitis_b IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
            UNION ALL
            SELECT 'bmi', SUM(CASE WHEN bmi IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
            UNION ALL
            SELECT 'polio', SUM(CASE WHEN polio IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
            UNION ALL
            SELECT 'total_expenditure', SUM(CASE WHEN total_expenditure IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
            UNION ALL
            SELECT 'diphtheria', SUM(CASE WHEN diphtheria IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
            UNION ALL
            SELECT 'gdp', SUM(CASE WHEN gdp IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
            UNION ALL
            SELECT 'population', SUM(CASE WHEN population IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
            UNION ALL
            SELECT 'thinness_1_19_years', SUM(CASE WHEN thinness_1_19_years IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
            UNION ALL
            SELECT 'thinness_5_9_years', SUM(CASE WHEN thinness_5_9_years IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
            UNION ALL
            SELECT 'income_composition_of_resources', SUM(CASE WHEN income_composition_of_resources IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
            UNION ALL
            SELECT 'schooling', SUM(CASE WHEN schooling IS NULL THEN 1 ELSE 0 END) FROM clean_life_expectancy
        )

        SELECT
            missing_counts.column_name,
            missing_counts.missing_count,
            ROUND(100.0 * missing_counts.missing_count / total.total_rows, 2) AS missing_percent
        FROM missing_counts
        CROSS JOIN total
        ORDER BY missing_count DESC;
        """

        export_query_to_csv(
            connection=conn,
            query=missingness_query,
            output_path=PROCESSED_DIR / "missingness_summary.csv"
        )

        print("Dashboard data export completed successfully.")

    finally:
        # Close the database connection when finished
        conn.close()


# This block runs only when the file is executed directly
if __name__ == "__main__":
    export_dashboard_data()