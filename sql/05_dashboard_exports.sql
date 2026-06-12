-- 05_dashboard_exports.sql
-- Purpose:
-- Dashboard-ready SQL queries
-- These outputs can be exported to CSV and used in Power BI

-- Status-level summary
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

-- Year-level summary
SELECT
    year,
    COUNT(*) AS num_records,
    ROUND(AVG(life_expectancy), 2) AS avg_life_expectancy,
    ROUND(MIN(life_expectancy), 2) AS min_life_expectancy,
    ROUND(MAX(life_expectancy), 2) AS max_life_expectancy
FROM clean_life_expectancy
GROUP BY year
ORDER BY year;

-- Country-level summary
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