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

