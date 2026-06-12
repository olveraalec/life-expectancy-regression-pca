-- 02_create_clean_view.sql
-- Purpose:
-- Create a clean SQL view with standardized column names
-- The raw CSV has spaces, capitalization issues, and special characters in column names
-- This view makes the dataset easier to query

CREATE OR REPLACE VIEW clean_life_expectancy AS
SELECT
    "Country" AS country,
    "Year" AS year,
    "Status" AS status,
    "Life expectancy" AS life_expectancy,
    "Adult Mortality" AS adult_mortality,
    "infant deaths" AS infant_deaths,
    "Alcohol" AS alcohol,
    "percentage expenditure" AS percentage_expenditure,
    "Hepatitis B" AS hepatitis_b,
    "Measles" AS measles,
    "BMI" AS bmi,
    "under-five deaths" AS under_five_deaths,
    "Polio" AS polio,
    "Total expenditure" AS total_expenditure,
    "Diphtheria" AS diphtheria,
    "HIV/AIDS" AS hiv_aids,
    "GDP" AS gdp,
    "Population" AS population,
    "thinness  1-19 years" AS thinness_1_19_years,
    "thinness 5-9 years" AS thinness_5_9_years,
    "Income composition of resources" AS income_composition_of_resources,
    "Schooling" AS schooling
FROM raw_life_expectancy;