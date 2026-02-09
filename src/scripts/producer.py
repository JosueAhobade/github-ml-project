import os
import json
import time
from datetime import datetime, timezone
from kafka import KafkaProducer
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =====================
# ENV
# =====================
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC")
KAFKA_API_KEY = os.getenv("KAFKA_API_KEY")
KAFKA_API_SECRET = os.getenv("KAFKA_API_SECRET")

GRAPHQL_URL = "https://api.github.com/graphql"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v4+json"
}

# =====================
# KAFKA PRODUCER
# =====================
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    security_protocol="SASL_SSL",
    sasl_mechanism="PLAIN",
    sasl_plain_username=KAFKA_API_KEY,
    sasl_plain_password=KAFKA_API_SECRET,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    linger_ms=10
)

# =====================
# SESSION REQUESTS AVEC RETRY
# =====================
session = requests.Session()
retry_strategy = Retry(
    total=5,  # retries
    backoff_factor=2,  # pause exponentielle
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)

def is_abandoned(issue, days_threshold=90):
    """
    Retourne True si l'issue est considérée comme abandonnée.
    """
    if issue["state"] == "CLOSED":
        return False  # une issue fermée n'est pas abandonnée

    last_update = datetime.fromisoformat(issue["updatedAt"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return (now - last_update).days > days_threshold

def compute_time_to_close(issue):
    """
    Retourne le temps de fermeture en jours pour une issue fermée.
    """
    if issue["state"] != "CLOSED" or not issue["closedAt"]:
        return None  # impossible de calculer pour une issue ouverte

    created = datetime.fromisoformat(issue["createdAt"].replace("Z", "+00:00"))
    closed = datetime.fromisoformat(issue["closedAt"].replace("Z", "+00:00"))
    delta = closed - created
    return delta.total_seconds() / 86400  # convertir en jours


def graphql(query, variables=None):
    try:
        r = session.post(
            GRAPHQL_URL,
            headers=HEADERS,
            json={"query": query, "variables": variables or {}},
            timeout=15  # timeout 15s
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Erreur API GitHub: {e}")
        return None

# =====================
# FETCH USERS
# =====================
def fetch_users(after=None, first=50):
    query = """
    query ($first:Int!, $after:String) {
      search(query: "type:user repos:>10 followers:>49", type: USER, first:$first, after:$after) {
        pageInfo { hasNextPage endCursor }
        nodes { ... on User { login } }
      }
    }
    """
    data = graphql(query, {"first": first, "after": after})
    if not data or "data" not in data:
        return None
    return data["data"]["search"]

# =====================
# FETCH REPOS
# =====================
def fetch_repos(user, after=None, first=10):
    query = """
    query ($login:String!, $first:Int!, $after:String) {
      user(login:$login) {
        repositories(first:$first, after:$after, orderBy:{field:STARGAZERS, direction:DESC}) {
          pageInfo { hasNextPage endCursor }
          nodes {
            name
            stargazerCount
            forkCount
            createdAt
            primaryLanguage { name }
          }
        }
      }
    }
    """
    data = graphql(query, {"login": user, "first": first, "after": after})
    if not data or "data" not in data or "user" not in data["data"]:
        return None
    return data["data"]["user"]["repositories"]

# =====================
# FETCH ISSUES
# =====================
def fetch_issues(owner, repo, after=None, first=20):
    query = """
    query ($owner:String!, $name:String!, $first:Int!, $after:String) {
      repository(owner:$owner, name:$name) {
        issues(first:$first, after:$after, states:[OPEN, CLOSED]) {
          pageInfo { hasNextPage endCursor }
          nodes {
            databaseId
            number
            title
            body
            state
            createdAt
            closedAt
            updatedAt
            comments { totalCount }
            labels(first:10) { nodes { name } }
          }
        }
      }
    }
    """
    data = graphql(query, {
        "owner": owner,
        "name": repo,
        "first": first,
        "after": after
    })
    if not data or "data" not in data or not data["data"]["repository"]:
        return None
    return data["data"]["repository"]["issues"]

# =====================
# PRODUCER PRINCIPAL
# =====================
# =====================
# PRODUCER PRINCIPAL
# =====================
def run():
    user_cursor = None

    while True:
        users = fetch_users(after=user_cursor)
        if not users:
            print("⚠️ Problème fetch_users, pause 60s...")
            time.sleep(60)
            continue

        for user in users["nodes"]:
            login = user["login"]
            print(f"👤 User: {login}")

            repo_cursor = None
            while True:
                repos = fetch_repos(login, after=repo_cursor)
                if not repos:
                    break

                for repo in repos["nodes"]:
                    repo_name = repo["name"]
                    print(f"  📦 Repo: {repo_name}")

                    issue_cursor = None
                    while True:
                        issues = fetch_issues(login, repo_name, after=issue_cursor)
                        if not issues:
                            break

                        for issue in issues["nodes"]:
                            if issue["databaseId"] is None:
                                continue

                            labels = [l["name"].lower() for l in issue["labels"]["nodes"]]

                            # ✅ calcul des nouvelles features
                            time_to_close = compute_time_to_close(issue)
                            abandoned = is_abandoned(issue)

                            message = {
                                "issue_id": issue["databaseId"],
                                "repo": f"{login}/{repo_name}",
                                "language": repo["primaryLanguage"]["name"] if repo["primaryLanguage"] else "unknown",
                                "stars": repo["stargazerCount"],
                                "forks": repo["forkCount"],
                                "issue_number": issue["number"],
                                "title_text": issue["title"] or "",   # <- nouveau champ
                                "body_text": issue["body"] or "",
                                "title_length": len(issue["title"] or ""),
                                "body_length": len(issue["body"] or ""),
                                "num_labels": len(labels),
                                "has_bug_label": int(any("bug" in l for l in labels)),
                                "contains_bug": int("bug" in ((issue["title"] or "") + (issue["body"] or "")).lower()),
                                "num_comments": issue["comments"]["totalCount"],
                                "state": issue["state"],
                                "created_at": issue["createdAt"],
                                "closed_at": issue["closedAt"],
                                "ingested_at": datetime.now(timezone.utc).isoformat(),
                                "repo_age_days": (datetime.fromisoformat(issue["createdAt"].replace("Z","")) - datetime.fromisoformat(repo["createdAt"].replace("Z",""))).days,
                                "created_weekday": datetime.fromisoformat(issue["createdAt"].replace("Z","")).weekday(),
                                "created_hour": datetime.fromisoformat(issue["createdAt"].replace("Z","")).hour,
                                "time_to_close": time_to_close,
                                "is_abandoned": int(abandoned),
                                "processed_for_model": 0,   # <- nouveau champ pour ML
                                "dataset_split": "none"     # <- nouveau champ pour train/val/test
                            }

                            producer.send(
                                KAFKA_TOPIC,
                                key=str(issue["databaseId"]).encode(),
                                value=message
                            )

                            # Pause pour éviter throttling
                            time.sleep(0.3)

                        if not issues["pageInfo"]["hasNextPage"]:
                            break
                        issue_cursor = issues["pageInfo"]["endCursor"]

                if not repos["pageInfo"]["hasNextPage"]:
                    break
                repo_cursor = repos["pageInfo"]["endCursor"]

        if not users["pageInfo"]["hasNextPage"]:
            break
        user_cursor = users["pageInfo"]["endCursor"]

        time.sleep(2)

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    run()
