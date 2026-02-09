import json
import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

JSONL_PATH = "issues_top20.jsonl_2"

conn = pymysql.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DB"),
    autocommit=False,      # mieux: on commit nous-mêmes en batch
    charset="utf8mb4",
)


INSERT_SQL = """\
INSERT INTO issues_history
(repo, issue_id, issue_number, pull_request, events, usernames, text_size, content, created_at)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON DUPLICATE KEY UPDATE
  issue_number = VALUES(issue_number),
  pull_request = VALUES(pull_request),
  events = VALUES(events),
  usernames = VALUES(usernames),
  text_size = VALUES(text_size),
  content = VALUES(content),
  created_at = VALUES(created_at);
"""

def to_mysql_datetime(dt_str: str):
    if not dt_str:
        return None
    s = dt_str.replace("T", " ").replace("Z", "")
    if "." in s:
        s = s.split(".", 1)[0]
    return s

def json_or_none(x):
    if x is None:
        return None
    return json.dumps(x, ensure_ascii=False)

def main():
    with conn.cursor() as cur:

        batch = []
        batch_size = 500

        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Ligne {line_no}: JSON invalide -> {e}")
                    continue

                repo = row.get("repo")
                issue_id = row.get("issue_id") or row.get("id")
                issue_number = row.get("issue_number") or row.get("number")

                pull_request = json_or_none(row.get("pull_request"))
                events = json_or_none(row.get("events"))
                usernames = json_or_none(row.get("usernames"))

                text_size = row.get("text_size")
                content = row.get("content")
                created_at = to_mysql_datetime(row.get("created_at"))

                if not repo:
                    print(f"Ligne {line_no}: repo manquant, skip")
                    continue

                batch.append((
                    repo, issue_id, issue_number,
                    pull_request, events, usernames,
                    text_size, content, created_at
                ))

                if len(batch) >= batch_size:
                    cur.executemany(INSERT_SQL, batch)
                    conn.commit()
                    batch.clear()

        if batch:
            cur.executemany(INSERT_SQL, batch)
            conn.commit()

    print("Import terminé.")

if __name__ == "__main__":
    try:
        main()
    finally:
        conn.close()