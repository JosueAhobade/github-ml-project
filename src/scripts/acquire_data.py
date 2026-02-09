#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

BASE_QUERY = """
SELECT issue_id, title_text, body_text, language, stars, forks, num_comments, 
       num_labels, contains_bug, repo_age_days, created_hour, time_to_close
FROM github_issues_v1
WHERE state = 'CLOSED'
  AND time_to_close > 0
  AND time_to_close < 8000
ORDER BY issue_id DESC
LIMIT %s;
"""

UPDATE_QUERY = """
UPDATE github_issues_v1
SET processed_for_model = 1
WHERE issue_id IN ({placeholders});
"""

def connect_mysql():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB"),
        autocommit=True
    )

def collect_to_csv(n_rows: int, out_path: str, mark_processed: bool = True) -> dict:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    db = connect_mysql()
    cursor = db.cursor()
    try:
        df = pd.read_sql(BASE_QUERY, db, params=(n_rows,))
        import csv

        df.to_csv(
            out_path,
            index=False,
            encoding="utf-8",
            escapechar="\\",              # IMPORTANT
            quoting=csv.QUOTE_MINIMAL,    # ou QUOTE_ALL si tu veux être safe
        )

        if mark_processed and len(df):
            issue_ids = df["issue_id"].tolist()
            placeholders = ",".join(["%s"] * len(issue_ids))
            update_query = UPDATE_QUERY.format(placeholders=placeholders)
            cursor.execute(update_query, issue_ids)

        return {"rows": int(len(df)), "out_path": out_path, "mark_processed": bool(mark_processed)}
    finally:
        cursor.close()
        db.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-rows", type=int, default=500000)
    p.add_argument("--out-path", type=str, default="data/raw/github_issues_collect.csv")
    p.add_argument("--mark-processed", action="store_true")
    p.add_argument("--no-mark-processed", dest="mark_processed", action="store_false")
    p.set_defaults(mark_processed=True)
    return p.parse_args()

def main():
    args = parse_args()
    res = collect_to_csv(args.n_rows, args.out_path, args.mark_processed)
    print(f"[OK] Saved {res['rows']} rows -> {res['out_path']}")

if __name__ == "__main__":
    main()