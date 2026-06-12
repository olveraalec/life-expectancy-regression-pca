"""
run_pipeline.py

Purpose:
Run the full end-to-end life expectancy project pipeline

This script:
1. Builds the DuckDB database from the raw CSV and SQL scripts
2. Exports dashboard-ready summary CSV files
3. Trains/evaluates regression models and exports model outputs

Run this file from the project root with:

    python run_pipeline.py
"""

# Import the database build function
# This creates or refreshes the DuckDB database using the SQL scripts
from src.build_database import build_database

# Import the dashboard data export function
# This creates CSV files for Power BI dashboard visuals
from src.export_dashboard_data import export_dashboard_data

# Import the model output export function
# This tunes models, creates model comparisons, and exports predictions
from src.export_model_outputs import export_model_outputs


def run_pipeline():
    """
    Run the complete project pipeline in the correct order
    """

    # Step 1:
    # Build or refresh the DuckDB database
    # This creates raw_life_expectancy, clean_life_expectancy, and modeling_life_expectancy
    print("Step 1: Building DuckDB database...")
    build_database()
    print("Step 1 complete.\n")

    # Step 2:
    # Export SQL-based dashboard summary files
    # These files are useful for Power BI
    print("Step 2: Exporting dashboard data...")
    export_dashboard_data()
    print("Step 2 complete.\n")

    # Step 3:
    # Train/evaluate models and export model outputs
    # This creates model_comparison.csv, alpha tuning files, and predictions.csv
    print("Step 3: Exporting model outputs...")
    export_model_outputs()
    print("Step 3 complete.\n")

    # Final completion message.
    print("Full pipeline completed successfully.")


# This block runs only when the script is executed directly
# It prevents the pipeline from automatically running if this file is imported elsewhere
if __name__ == "__main__":
    run_pipeline()