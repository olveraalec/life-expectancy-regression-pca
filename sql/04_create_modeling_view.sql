-- 04_create_modeling_view.sql
-- Purpose:
-- Create the official SQL modeling view
-- SQL defines the modeling dataset
-- Python will handle imputation, scaling, PCA, regularization, and model training

CREATE OR REPLACE VIEW modeling_life_expectancy AS
SELECT
    country,
    year,
    status,
    life_expectancy,
    adult_mortality,
    infant_deaths,
    alcohol,
    percentage_expenditure,
    hepatitis_b,
    measles,
    bmi,
    under_five_deaths,
    polio,
    total_expenditure,
    diphtheria,
    hiv_aids,
    gdp,
    population,
    thinness_1_19_years,
    thinness_5_9_years,
    income_composition_of_resources,
    schooling
FROM clean_life_expectancy
WHERE life_expectancy IS NOT NULL;