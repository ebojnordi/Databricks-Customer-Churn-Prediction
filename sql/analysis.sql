-- Databricks Customer Churn Analysis
-- Business-oriented analysis using the Gold feature table

-- 1. Overall churn rate
SELECT
    COUNT(*) AS total_customers,
    SUM(Churn) AS churned_customers,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct
FROM churn_prediction.gold.churn_features;


-- 2. Churn rate by contract type
SELECT
    Contract,
    COUNT(*) AS customers,
    SUM(Churn) AS churned_customers,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct
FROM churn_prediction.gold.churn_features
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- 3. Churn rate by internet service
SELECT
    InternetService,
    COUNT(*) AS customers,
    SUM(Churn) AS churned_customers,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct
FROM churn_prediction.gold.churn_features
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;


-- 4. Churn rate by tenure group
SELECT
    CASE
        WHEN tenure < 12 THEN 'Less than 1 year'
        WHEN tenure < 24 THEN '1-2 years'
        WHEN tenure < 48 THEN '2-4 years'
        ELSE '4+ years'
    END AS tenure_group,
    COUNT(*) AS customers,
    SUM(Churn) AS churned_customers,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct
FROM churn_prediction.gold.churn_features
GROUP BY
    CASE
        WHEN tenure < 12 THEN 'Less than 1 year'
        WHEN tenure < 24 THEN '1-2 years'
        WHEN tenure < 48 THEN '2-4 years'
        ELSE '4+ years'
    END
ORDER BY churn_rate_pct DESC;


-- 5. Monthly charges and churn
SELECT
    Churn,
    COUNT(*) AS customers,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(AVG(tenure), 2) AS avg_tenure
FROM churn_prediction.gold.churn_features
GROUP BY Churn
ORDER BY Churn;
