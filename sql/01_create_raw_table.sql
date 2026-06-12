-- 01_create_raw_table.sql
-- Purpose:
-- Load the raw life expectancy CSV into a DuckDB table
-- This creates the starting database table for the project

CREATE OR REPLACE TABLE raw_life_expectancy AS
SELECT *
FROM read_csv_auto('data/raw/life_expectancy.csv', header=True);