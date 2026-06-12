-- 03_data_quality_checks.sql
-- Purpose:
-- Run SQL-based data quality checks for the life expectancy dataset
-- These queries inspect row counts, year coverage, development status groups
-- and missing values by variable

-- Dataset overview
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT country) AS unique_countries,
    MIN(year) AS first_year,
    MAX(year) AS last_year,
    COUNT(DISTINCT status) AS unique_status_values
FROM clean_life_expectancy;

-- Life expectancy by development status
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

-- Average life expectancy by year
SELECT
    year,
    COUNT(*) AS num_records,
    ROUND(AVG(life_expectancy), 2) AS avg_life_expectancy,
    ROUND(MIN(life_expectancy), 2) AS min_life_expectancy,
    ROUND(MAX(life_expectancy), 2) AS max_life_expectancy
FROM clean_life_expectancy
GROUP BY year
ORDER BY year;

-- Missingness summary
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