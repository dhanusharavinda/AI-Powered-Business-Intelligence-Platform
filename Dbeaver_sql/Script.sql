create schema warehouse;

CREATE TABLE warehouse.dim_accounts (
    account_id VARCHAR PRIMARY KEY,
    account_name VARCHAR,
    industry VARCHAR,
    country VARCHAR,
    signup_date DATE,
    referral_source VARCHAR,
    plan_tier VARCHAR,
    seats INTEGER,
    is_trial BOOLEAN,
    churn_flag BOOLEAN
);

CREATE TABLE warehouse.fact_subscriptions (
    subscription_id VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    start_date DATE,
    end_date DATE,
    plan_tier VARCHAR,
    seats INTEGER,
    mrr_amount NUMERIC,
    arr_amount NUMERIC,
    is_trial BOOLEAN,
    upgrade_flag BOOLEAN,
    downgrade_flag BOOLEAN,
    churn_flag BOOLEAN,
    billing_frequency VARCHAR,
    auto_renew_flag BOOLEAN,
    is_active BOOLEAN,
    CONSTRAINT fk_subscription_account
        FOREIGN KEY (account_id)
        REFERENCES warehouse.dim_accounts(account_id)
);

CREATE TABLE warehouse.fact_feature_usage (
    usage_id VARCHAR PRIMARY KEY,
    subscription_id VARCHAR NOT NULL,
    usage_date DATE,
    feature_name VARCHAR,
    usage_count INTEGER,
    usage_duration_secs NUMERIC,
    error_count INTEGER,
    is_beta_feature BOOLEAN,
    CONSTRAINT fk_usage_subscription
        FOREIGN KEY (subscription_id)
        REFERENCES warehouse.fact_subscriptions(subscription_id)
);

CREATE TABLE warehouse.fact_support (
    ticket_id VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    submitted_at TIMESTAMP,
    closed_at TIMESTAMP,
    resolution_time_hours NUMERIC,
    priority VARCHAR,
    first_response_time_minutes NUMERIC,
    satisfaction_score NUMERIC,
    escalation_flag BOOLEAN,
    CONSTRAINT fk_support_account
        FOREIGN KEY (account_id)
        REFERENCES warehouse.dim_accounts(account_id)
);

CREATE TABLE warehouse.fact_churn (
    churn_event_id VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    churn_date DATE,
    reason_code VARCHAR,
    refund_amount_usd NUMERIC,
    preceding_upgrade_flag BOOLEAN,
    preceding_downgrade_flag BOOLEAN,
    is_reactivation BOOLEAN,
    feedback_text TEXT,
    CONSTRAINT fk_churn_account
        FOREIGN KEY (account_id)
        REFERENCES warehouse.dim_accounts(account_id)
);



 #creating views
 
 CREATE OR REPLACE VIEW warehouse.monthly_mrr AS
SELECT
    DATE_TRUNC('month', start_date) AS month,
    SUM(mrr_amount) AS total_mrr
FROM warehouse.fact_subscriptions
GROUP BY 1
ORDER BY 1;

CREATE OR REPLACE VIEW warehouse.revenue_by_plan AS
SELECT
    plan_tier,
    SUM(mrr_amount) AS total_mrr,
    SUM(arr_amount) AS total_arr,
    COUNT(*) AS subscription_count
FROM warehouse.fact_subscriptions
GROUP BY plan_tier
ORDER BY total_mrr DESC;

CREATE OR REPLACE VIEW warehouse.revenue_by_region AS
SELECT
    da.country,
    SUM(fs.mrr_amount) AS total_mrr,
    SUM(fs.arr_amount) AS total_arr
FROM warehouse.fact_subscriptions fs
JOIN warehouse.dim_accounts da
  ON fs.account_id = da.account_id
GROUP BY da.country
ORDER BY total_mrr DESC;

CREATE OR REPLACE VIEW warehouse.churned_mrr AS
SELECT
    DATE_TRUNC('month', start_date) AS month,
    SUM(mrr_amount) AS churned_mrr
FROM warehouse.fact_subscriptions
WHERE churn_flag = TRUE
GROUP BY 1
ORDER BY 1;

CREATE OR REPLACE VIEW warehouse.monthly_churn_rate AS
WITH monthly_totals AS (
    SELECT
        DATE_TRUNC('month', start_date) AS month,
        COUNT(*) AS total_subscriptions
    FROM warehouse.fact_subscriptions
    GROUP BY 1
),
monthly_churned AS (
    SELECT
        DATE_TRUNC('month', start_date) AS month,
        COUNT(*) AS churned_subscriptions
    FROM warehouse.fact_subscriptions
    WHERE churn_flag = TRUE
    GROUP BY 1
)
SELECT
    mt.month,
    mt.total_subscriptions,
    COALESCE(mc.churned_subscriptions, 0) AS churned_subscriptions,
    ROUND(
        COALESCE(mc.churned_subscriptions, 0)::numeric
        / NULLIF(mt.total_subscriptions, 0),
        4
    ) AS churn_rate
FROM monthly_totals mt
LEFT JOIN monthly_churned mc
    ON mt.month = mc.month
ORDER BY mt.month;

select * from warehouse.monthly_churn_rate


CREATE OR REPLACE VIEW warehouse.churn_by_plan AS
SELECT
    plan_tier,
    COUNT(*) AS total_subscriptions,
    COUNT(*) FILTER (WHERE churn_flag = TRUE) AS churned_count,
    ROUND(
        COUNT(*) FILTER (WHERE churn_flag = TRUE)::numeric
        / NULLIF(COUNT(*), 0),
        4
    ) AS churn_rate
FROM warehouse.fact_subscriptions
GROUP BY plan_tier
ORDER BY churn_rate DESC;

select * from warehouse.churn_by_plan

CREATE OR REPLACE VIEW warehouse.churn_by_industry AS
SELECT
    da.industry,
    COUNT(*) AS total_subscriptions,
    COUNT(*) FILTER (WHERE fs.churn_flag = TRUE) AS churned_count,
    ROUND(
        COUNT(*) FILTER (WHERE fs.churn_flag = TRUE)::numeric
        / NULLIF(COUNT(*), 0),
        4
    ) AS churn_rate
FROM warehouse.fact_subscriptions fs
JOIN warehouse.dim_accounts da
    ON fs.account_id = da.account_id
GROUP BY da.industry
ORDER BY churn_rate DESC;

select * from warehouse.churn_by_industry

CREATE OR REPLACE VIEW warehouse.avg_subscription_duration AS
SELECT
    AVG(end_date - start_date) AS avg_duration_days
FROM warehouse.fact_subscriptions
WHERE churn_flag = TRUE
AND end_date IS NOT NULL;

select * from warehouse.avg_subscription_duration

CREATE OR REPLACE VIEW warehouse.upgrade_before_churn_rate AS
WITH churned_accounts AS (
    SELECT account_id
    FROM warehouse.fact_subscriptions
    WHERE churn_flag = TRUE
),
multi_plan_accounts AS (
    SELECT account_id
    FROM warehouse.fact_subscriptions
    GROUP BY account_id
    HAVING COUNT(DISTINCT plan_tier) > 1
)
SELECT
    COUNT(*) FILTER (WHERE mp.account_id IS NOT NULL)::numeric
    / NULLIF(COUNT(*), 0) AS upgrade_before_churn_rate
FROM churned_accounts ca
LEFT JOIN multi_plan_accounts mp
    ON ca.account_id = mp.account_id;

select * from warehouse.upgrade_before_churn_rate

CREATE OR REPLACE VIEW warehouse.support_churn_correlation AS
WITH support_counts AS (
    SELECT
        account_id,
        COUNT(*) AS ticket_count
    FROM warehouse.fact_support
    GROUP BY account_id
)
SELECT
    CASE
        WHEN sc.ticket_count >= 5 THEN 'high_support'
        ELSE 'low_support'
    END AS support_bucket,
    COUNT(*) AS total_accounts,
    COUNT(*) FILTER (WHERE fs.churn_flag = TRUE) AS churned_accounts,
    ROUND(
        COUNT(*) FILTER (WHERE fs.churn_flag = TRUE)::numeric
        / NULLIF(COUNT(*), 0),
        4
    ) AS churn_rate
FROM warehouse.fact_subscriptions fs
LEFT JOIN support_counts sc
    ON fs.account_id = sc.account_id
GROUP BY support_bucket
ORDER BY churn_rate DESC;

SELECT * FROM warehouse.support_churn_correlation;


SELECT
    table_name AS view_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'warehouse'
AND table_name IN (
    'monthly_churn_rate',
    'churn_by_plan',
    'churn_by_industry',
    'avg_subscription_duration',
    'upgrade_before_churn_rate',
    'support_churn_correlation'
)
ORDER BY table_name, ordinal_position;


