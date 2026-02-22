import pandas as pd
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ---------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_SCHEMA = os.getenv("DB_SCHEMA")
print("DB_SCHEMA from env:", DB_SCHEMA)

# ---------------------------------------
# CREATE DATABASE CONNECTION
# ---------------------------------------
engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ---------------------------------------
# EXTRACT (CLEANED FILES)
# ---------------------------------------
def extract():
    print("Extracting cleaned datasets...")

    base_path = "Saas_Synthetic_Dataset/data/cleaned/"

    return {
        "accounts": pd.read_csv(base_path + "ravenstack_accounts_enriched_cleaned.csv"),
        "subscriptions": pd.read_csv(base_path + "ravenstack_subscriptions_cleaned.csv"),
        "feature_usage": pd.read_csv(base_path + "ravenstack_feature_usage_enriched_cleaned.csv"),
        "support": pd.read_csv(base_path + "ravenstack_support_tickets_cleaned.csv"),
        "churn": pd.read_csv(base_path + "ravenstack_churn_events_cleaned.csv")
    }


    
#TRANSFORM wasn't necessary for this project.
# ---------------------------------------
# LOAD (FULL REFRESH)
# ---------------------------------------
def load(data):
    print("Performing full refresh load...")

    with engine.begin() as conn:
        # Truncate fact tables first (due to FK constraints)
        conn.execute(text(f"TRUNCATE TABLE {DB_SCHEMA}.fact_churn CASCADE"))
        conn.execute(text(f"TRUNCATE TABLE {DB_SCHEMA}.fact_support CASCADE"))
        conn.execute(text(f"TRUNCATE TABLE {DB_SCHEMA}.fact_feature_usage CASCADE"))
        conn.execute(text(f"TRUNCATE TABLE {DB_SCHEMA}.fact_subscriptions CASCADE"))
        conn.execute(text(f"TRUNCATE TABLE {DB_SCHEMA}.dim_accounts CASCADE"))

        data["feature_usage"] = (
    data["feature_usage"]
    .drop_duplicates(subset=["usage_id"])
)

    # Load dimension first
    data["accounts"].to_sql(
        "dim_accounts",
        engine,
        schema=DB_SCHEMA,
        if_exists="append",
        index=False
    )

    # Load fact tables
    data["subscriptions"].to_sql(
        "fact_subscriptions",
        engine,
        schema=DB_SCHEMA,
        if_exists="append",
        index=False
    )

    data["feature_usage"].to_sql(
        "fact_feature_usage",
        engine,
        schema=DB_SCHEMA,
        if_exists="append",
        index=False
    )

    data["support"].to_sql(
        "fact_support",
        engine,
        schema=DB_SCHEMA,
        if_exists="append",
        index=False
    )

    data["churn"].to_sql(
        "fact_churn",
        engine,
        schema=DB_SCHEMA,
        if_exists="append",
        index=False
    )

# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    try:
        data = extract()
        load(data)
        print("ETL completed successfully.")
    except Exception as e:
        print("ETL failed:", e)

if __name__ == "__main__":
    main()