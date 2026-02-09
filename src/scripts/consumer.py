import json
import os
from kafka import KafkaConsumer
import mysql.connector
from mysql.connector import OperationalError
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# =====================
# Kafka
# =====================
consumer = KafkaConsumer(
    os.getenv("KAFKA_TOPIC"),
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
    security_protocol="SASL_SSL",
    sasl_mechanism="PLAIN",
    sasl_plain_username=os.getenv("KAFKA_API_KEY"),
    sasl_plain_password=os.getenv("KAFKA_API_SECRET"),
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="mysql-consumer-issues-v1"
)

# =====================
# MySQL
# =====================
def connect_mysql():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB"),
        autocommit=True
    )

db = connect_mysql()
cursor = db.cursor()

# =====================
# INSERT QUERY
# =====================
INSERT_QUERY = """
INSERT INTO github_issues_v1 (
    issue_id, repo, language, stars, forks, issue_number,
    title_text, body_text, title_length, body_length,
    num_labels, has_bug_label, contains_bug, num_comments,
    state, created_at, closed_at, ingested_at,
    repo_age_days, created_weekday, created_hour,
    time_to_close, is_abandoned,
    processed_for_model, dataset_split
)
VALUES (%s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s)
ON DUPLICATE KEY UPDATE
    ingested_at = VALUES(ingested_at),
    time_to_close = VALUES(time_to_close),
    is_abandoned = VALUES(is_abandoned),
    processed_for_model = VALUES(processed_for_model),
    dataset_split = VALUES(dataset_split);
"""

# =====================
# SAFE INSERT
# =====================
def safe_insert(issue):
    global db, cursor

    while True:
        try:
            cursor.execute(
                INSERT_QUERY,
                (
                    issue["issue_id"],
                    issue["repo"],
                    issue["language"],
                    issue["stars"],
                    issue["forks"],
                    issue["issue_number"],
                    issue["title_text"],
                    issue["body_text"],
                    issue["title_length"],
                    issue["body_length"],
                    issue["num_labels"],
                    issue["has_bug_label"],
                    issue["contains_bug"],
                    issue["num_comments"],
                    issue["state"],
                    issue["created_at"].replace("T", " ").replace("Z", ""),
                    issue["closed_at"].replace("T", " ").replace("Z", "") if issue["closed_at"] else None,
                    issue["ingested_at"].replace("T", " ").replace("Z", ""),
                    issue["repo_age_days"],
                    issue["created_weekday"],
                    issue["created_hour"],
                    issue["time_to_close"],
                    issue["is_abandoned"],
                    issue["processed_for_model"],
                    issue["dataset_split"]
                )
            )
            break

        except OperationalError as e:
            print(f"⚠️ MySQL connection lost, reconnecting... {e}")
            db.reconnect(attempts=5, delay=5)
            cursor = db.cursor()

        except mysql.connector.Error as e:
            print(f"❌ MySQL error: {e} | issue_id={issue['issue_id']}")
            break

# =====================
# MAIN LOOP
# =====================
for msg in consumer:
    issue = msg.value
    safe_insert(issue)
    print(f"✅ Issue {issue['issue_id']} ingérée")
