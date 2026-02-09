# app.py
# --------------------------------------------
# GitHub Issues Dashboard FINAL
# - Monitoring + filtres + KPIs + charts
# - Onglet Prédiction (API FastAPI /predict) conforme à IssueInput
# - Onglet Réentrainement: /train + /train/status/{job_id} + /reload-model + /model/info
#
# Requirements:
#   pip install streamlit pandas mysql-connector-python plotly requests
#
# Secrets (.streamlit/secrets.toml) recommandés:
#   DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT
#   API_BASE (ex: http://localhost:8000) ou PREDICT_URL/TRAIN_URL
# --------------------------------------------

import time
import pandas as pd
import mysql.connector
import plotly.express as px
import requests
import streamlit as st
import os
from dotenv import load_dotenv
import csv

load_dotenv()  # charge .env à la racine


# =========================
# CONFIG UI
# =========================
st.set_page_config(page_title="GitHub Big Data Monitor", layout="wide")

DAYS_MAP = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}


# =========================
# API CONFIG
# =========================
# Support 2 modes:
# 1) API_BASE = http://localhost:8000
# 2) PREDICT_URL / TRAIN_URL (legacy)

def cfg(key: str, default=None):
    return os.getenv(key, default)


API_BASE = cfg("API_BASE", "http://api:8000").rstrip("/")
API_TOKEN = cfg("API_TOKEN", "")

PREDICT_URL = f"{API_BASE}/predict"
TRAIN_URL = f"{API_BASE}/train"
STATUS_URL = f"{API_BASE}/train/status"
RELOAD_URL = f"{API_BASE}/reload-model"
MODEL_INFO_URL = f"{API_BASE}/model/info"

def api_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


# =========================
# DB LOAD
# =========================
@st.cache_data(ttl=30)
def load_data() -> pd.DataFrame:
    """
    Charge les données depuis MySQL.
    Essaie de récupérer aussi les colonnes nécessaires pour la prédiction:
      title_text, body_text, forks, num_comments, num_labels, created_hour, issue_id
    Sinon fallback minimal.
    """
    config = {
        "host": cfg("MYSQL_HOST"),
        "user": cfg("MYSQL_USER"),
        "password": cfg("MYSQL_PASSWORD"),
        "database": cfg("MYSQL_DB"),
        "port": int(cfg("MYSQL_PORT", 3306)),
    }

    query_full = """
    SELECT
        issue_id,
        repo,
        title_text,
        body_text,
        language,
        stars,
        forks,
        num_comments,
        num_labels,
        contains_bug,
        repo_age_days,
        created_hour,
        created_weekday,
        state,
        time_to_close,
        is_abandoned,
        has_bug_label,
        ingested_at
    FROM github_issues_v1
    LiMIT 300000
    """

    query_min = """
    SELECT
        repo, language, stars, state, time_to_close,
        is_abandoned, has_bug_label, contains_bug,
        created_weekday, repo_age_days, ingested_at
    FROM github_issues_v1
    """

    try:
        conn = mysql.connector.connect(**config)
        try:
            df = pd.read_sql(query_full, conn)
            if "ingested_at" in df.columns:
                df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce")
                df["ingest_day"] = df["ingested_at"].dt.date.astype(str)   # "YYYY-MM-DD"
                df["ingest_hour"] = df["ingested_at"].dt.hour
        except Exception:
            df = pd.read_sql(query_min, conn)
            if "ingested_at" in df.columns:
                df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce")
                df["ingest_day"] = df["ingested_at"].dt.date.astype(str)   # "YYYY-MM-DD"
                df["ingest_hour"] = df["ingested_at"].dt.hour
        finally:
            conn.close()

        # cleaning
        if "created_weekday" in df.columns:
            df["Day Name"] = df["created_weekday"].map(DAYS_MAP)
            df["Day Name"] = pd.Categorical(df["Day Name"], categories=list(DAYS_MAP.values()), ordered=True)

        for col in ["stars", "forks", "num_comments", "num_labels", "contains_bug", "repo_age_days", "created_hour"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        for col in ["is_abandoned", "has_bug_label"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        if "time_to_close" in df.columns:
            df["time_to_close"] = pd.to_numeric(df["time_to_close"], errors="coerce")

        for col in ["title_text", "body_text", "repo", "language", "state"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        return df

    except Exception as e:
        st.error(f"Erreur SQL : {e}")
        return pd.DataFrame()


# =========================
# HELPERS
# =========================
def fmt_hours_to_days(h):
    if h is None or pd.isna(h):
        return "—"
    h = float(h)
    return f"{h/24:.1f} j" if h >= 24 else f"{h:.1f} h"

def has_predict_cols(df: pd.DataFrame) -> bool:
    needed = {
        "title_text", "body_text", "language", "stars", "forks",
        "num_comments", "num_labels", "contains_bug", "repo_age_days", "created_hour"
    }
    return needed.issubset(set(df.columns))


# =========================
# APP
# =========================
st.title("GitHub Issues Big Data")
st.markdown("Monitoring des issues + prédiction + réentrainement.")

with st.spinner("Chargement des données..."):
    df = load_data()

if df.empty:
    st.warning("Aucune donnée trouvée. Vérifier la connexion et la table.")
    st.stop()


# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filtres")

langs = sorted(df["language"].dropna().unique()) if "language" in df.columns else []
states = sorted(df["state"].dropna().unique()) if "state" in df.columns else []

# Langages "célèbres" à pré-sélectionner
DEFAULT_LANGS = ["Java", "Python", "JavaScript"]

# Garder uniquement ceux présents dans les données
default_langs = [l for l in DEFAULT_LANGS if l in langs]

sel_langs = st.sidebar.multiselect(
    "Langages",
    langs,
    default=default_langs
)
sel_states = st.sidebar.multiselect("State", states, default=states)

stars_range = None
if "stars" in df.columns and len(df):
    min_stars, max_stars = int(df["stars"].min()), int(df["stars"].max())
    stars_range = st.sidebar.slider("Stars (repo)", min_value=min_stars, max_value=max_stars, value=(min_stars, max_stars))

abandon_filter = st.sidebar.radio("Abandon", ["Tous", "Abandonnées", "Non abandonnées"], index=0)
bug_filter = st.sidebar.radio("Bug", ["Tous", "Label bug", "Contient 'bug'", "Label OU texte"], index=0)

fdf = df.copy()

if sel_langs and "language" in fdf.columns:
    fdf = fdf[fdf["language"].isin(sel_langs)]
if sel_states and "state" in fdf.columns:
    fdf = fdf[fdf["state"].isin(sel_states)]
if stars_range and "stars" in fdf.columns:
    fdf = fdf[(fdf["stars"] >= stars_range[0]) & (fdf["stars"] <= stars_range[1])]

if "is_abandoned" in fdf.columns:
    if abandon_filter == "Abandonnées":
        fdf = fdf[fdf["is_abandoned"] == 1]
    elif abandon_filter == "Non abandonnées":
        fdf = fdf[fdf["is_abandoned"] == 0]

if bug_filter != "Tous":
    if bug_filter == "Label bug" and "has_bug_label" in fdf.columns:
        fdf = fdf[fdf["has_bug_label"] == 1]
    elif bug_filter == "Contient 'bug'" and "contains_bug" in fdf.columns:
        fdf = fdf[fdf["contains_bug"] == 1]
    elif bug_filter == "Label OU texte" and {"has_bug_label", "contains_bug"}.issubset(fdf.columns):
        fdf = fdf[(fdf["has_bug_label"] == 1) | (fdf["contains_bug"] == 1)]

ingest_days = []
if "ingest_day" in df.columns:
    ingest_days = sorted(df["ingest_day"].dropna().unique(), reverse=True)

sel_ingest_days = st.sidebar.multiselect(
    "Jour d'ingestion",
    options=ingest_days,
    # default=ingest_days[:3] if len(ingest_days) > 3 else ingest_days
)

if sel_ingest_days and "ingest_day" in fdf.columns:
    fdf = fdf[fdf["ingest_day"].isin(sel_ingest_days)]


# =========================
# KPI
# =========================
total = len(fdf)
abandoned = int(fdf["is_abandoned"].sum()) if "is_abandoned" in fdf.columns else 0
open_issues = int((fdf["state"] == "OPEN").sum()) if "state" in fdf.columns else 0
closed_issues = int((fdf["state"] == "CLOSED").sum()) if "state" in fdf.columns else 0

pct_abandoned = (abandoned / total * 100) if total else 0
pct_closed = (closed_issues / total * 100) if total else 0

closed_df = (
    fdf[fdf["state"] == "CLOSED"].dropna(subset=["time_to_close"])
    if {"state", "time_to_close"}.issubset(fdf.columns) else pd.DataFrame()
)
median_time_h = closed_df["time_to_close"].median() if len(closed_df) else None
p90_time_h = closed_df["time_to_close"].quantile(0.90) if len(closed_df) else None

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Issues (filtrées)", f"{total:,}".replace(",", " "))
col2.metric("OPEN", f"{open_issues:,}".replace(",", " "))
col3.metric("CLOSED", f"{closed_issues:,}".replace(",", " "), f"{pct_closed:.1f}%")
col4.metric("Abandonnées", f"{abandoned:,}".replace(",", " "), f"{pct_abandoned:.1f}%")
col5.metric("Temps médian (CLOSED)", fmt_hours_to_days(median_time_h))

if "ingested_at" in df.columns:
    # s'assurer que c'est bien un datetime
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce")

    now = pd.Timestamp.now()

    last_1d = df[df["ingested_at"] >= (now - pd.Timedelta(days=1))]
    last_3d = df[df["ingested_at"] >= (now - pd.Timedelta(days=3))]

    k1, k2, k3 = st.columns(3)
    k1.metric("🟢 Ingest (24h)", f"{len(last_1d):,}".replace(",", " "))
    k2.metric("🟡 Ingest (3 jours)", f"{len(last_3d):,}".replace(",", " "))
    k3.metric(
        "Dernière ingestion",
        df["ingested_at"].max().strftime("%Y-%m-%d %H:%M:%S")
        if df["ingested_at"].notna().any()
        else "—"
    )
else:
    st.info("Colonne `ingested_at` absente → impossible d'afficher l'ingestion.")

st.caption(f"P90 temps de résolution (CLOSED) : {fmt_hours_to_days(p90_time_h)}")

st.divider()


# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Vue Globale", "Analyse de l'Abandon", "Impact Popularité", "Prédictions", "Réentrainement"]
)

# -------------------------
# TAB 1
# -------------------------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Activité par langage")
        if "language" in fdf.columns:
            lang_counts = fdf["language"].value_counts().reset_index()
            lang_counts.columns = ["language", "count"]
            fig_lang = px.bar(lang_counts, x="language", y="count", title="Volume d'issues par langage")
            st.plotly_chart(fig_lang, use_container_width=True)
        else:
            st.info("Colonne 'language' absente.")

    with c2:
        st.subheader("Label bug vs texte 'bug'")
        if {"has_bug_label", "contains_bug"}.issubset(fdf.columns):
            bug_stats = pd.DataFrame({
                "Type": ["Label officiel 'Bug'", "Mot-clé 'bug' dans texte"],
                "Count": [int(fdf["has_bug_label"].sum()), int(fdf["contains_bug"].sum())],
            })
            fig_bug = px.bar(bug_stats, x="Type", y="Count", color="Type",
                             title="Les développeurs oublient-ils de taguer les bugs ?")
            st.plotly_chart(fig_bug, use_container_width=True)
        else:
            st.info("Colonnes 'has_bug_label' / 'contains_bug' absentes.")
    

# -------------------------
# TAB 2
# -------------------------
with tab2:
    st.subheader("Abandon par langage")
    if {"language", "is_abandoned"}.issubset(fdf.columns):
        abandon_by_lang = fdf.groupby("language")["is_abandoned"].mean().mul(100).reset_index()
        fig_ab = px.bar(
            abandon_by_lang.sort_values("is_abandoned"),
            x="language", y="is_abandoned",
            labels={"is_abandoned": "% Abandon"},
            title="Pourcentage d'issues abandonnées par langage",
        )
        st.plotly_chart(fig_ab, use_container_width=True)
    else:
        st.info("Colonnes nécessaires absentes (language/is_abandoned).")

    st.subheader("Abandon selon le jour de création")
    if {"Day Name", "is_abandoned"}.issubset(fdf.columns):
        by_day = fdf.groupby("Day Name")["is_abandoned"].mean().mul(100).reset_index()
        fig_day = px.line(by_day, x="Day Name", y="is_abandoned", markers=True, title="% abandon vs jour")
        st.plotly_chart(fig_day, use_container_width=True)
    else:
        st.info("Colonnes nécessaires absentes (created_weekday / Day Name / is_abandoned).")

# -------------------------
# TAB 3
# -------------------------
with tab3:
    st.subheader("Stars vs temps de résolution")
    st.caption("Échelles log.")

    if {"state", "time_to_close", "stars", "language", "repo"}.issubset(fdf.columns):
        closed_sample = fdf[fdf["state"] == "CLOSED"].dropna(subset=["time_to_close"]).copy()
        if len(closed_sample) > 5000:
            closed_sample = closed_sample.sample(5000, random_state=42)

        closed_sample["stars_plot"] = closed_sample["stars"].clip(lower=1)
        closed_sample["ttc_plot"] = closed_sample["time_to_close"].clip(lower=1e-6)

        fig_corr = px.scatter(
            closed_sample,
            x="stars_plot",
            y="ttc_plot",
            color="language",
            log_x=True,
            log_y=True,
            hover_data=["repo", "stars", "time_to_close"],
            title="Popularité (stars) vs temps de fermeture",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Colonnes nécessaires absentes (state/time_to_close/stars/language/repo).")

# -------------------------
# TAB 4 : PREDICTION
# -------------------------
with tab4:
    st.subheader("Prédiction")
    st.caption("Sélectionne une issue → champs pré-remplis → appel API → label (Flash/Day/Slow).")

    if not has_predict_cols(df):
        st.warning(
            "Ta table SQL ne contient pas toutes les colonnes attendues par l'API "
            "(title_text, body_text, forks, num_comments, num_labels, created_hour...). "
            "Le formulaire reste utilisable, mais on devra remplir certains champs manuellement."
        )
    # --- mini dataset pour la prédiction (simple) ---
    N_PRED = 10

    if "issue_id" in fdf.columns:
        pred_df = fdf.sort_values("issue_id", ascending=False).head(N_PRED).copy()
    else:
        pred_df = fdf.head(N_PRED).copy()

    # Choix d’issue : si issue_id dispo, c’est mieux, sinon index
    if "issue_id" in df.columns:
        ids = pred_df["issue_id"].tolist()
        if len(ids) == 0:
            st.info("Aucune issue dans le filtre courant.")
            st.stop()
        selected_id = st.selectbox("issue_id", ids, key="pred_issue_id")
        row = pred_df[pred_df["issue_id"] == selected_id].iloc[0].to_dict()
    else:
        max_idx = len(pred_df) - 1
        idx = st.number_input("Index de l'issue (fdf)", min_value=0, max_value=max_idx, value=0, step=1)
        row = pred_df.iloc[int(idx)].to_dict()
    def get_str(key, default=""):
        v = row.get(key, default)
        return "" if v is None else str(v)

    def get_int(key, default=0):
        v = row.get(key, default)
        try:
            return int(v)
        except Exception:
            return int(default)

    with st.form("predict_form", clear_on_submit=False):
        st.write("Champs envoyés à l'API")

        c1, c2 = st.columns(2)
        with c1:
            title = st.text_input("title", value=get_str("title_text", get_str("title")))
            body = st.text_area("body", value=get_str("body_text", get_str("body")), height=160)
            language = st.text_input("language", value=get_str("language"))

        with c2:
            stars = st.number_input("stars", min_value=0, value=get_int("stars"))
            forks = st.number_input("forks", min_value=0, value=get_int("forks"))
            num_comments = st.number_input("num_comments", min_value=0, value=get_int("num_comments"))
            num_labels = st.number_input("num_labels", min_value=0, value=get_int("num_labels"))
            contains_bug = st.selectbox("contains_bug", options=[0, 1], index=1 if get_int("contains_bug") == 1 else 0)
            repo_age_days = st.number_input("repo_age_days", min_value=0, value=get_int("repo_age_days"))
            created_hour = st.number_input("created_hour (0-23)", min_value=0, max_value=23,
                                           value=min(max(get_int("created_hour"), 0), 23))

        submitted = st.form_submit_button("🚀 Prédire")

    if submitted:
        if not title.strip():
            st.error("Le champ 'title' est obligatoire (ton API le demande).")
        elif not language.strip():
            st.error("Le champ 'language' est obligatoire.")
        else:
            payload = {
                "title": title,
                "body": body or "",
                "language": language,
                "stars": int(stars),
                "forks": int(forks),
                "num_comments": int(num_comments),
                "num_labels": int(num_labels),
                "contains_bug": int(contains_bug),
                "repo_age_days": int(repo_age_days),
                "created_hour": int(created_hour),
            }

            with st.spinner("Appel de l'API de prédiction..."):
                try:
                    r = requests.post(PREDICT_URL, json=payload, headers=api_headers(), timeout=20)
                    if r.status_code >= 400:
                        st.error(f"Erreur API ({r.status_code}) : {r.text}")
                    else:
                        data = r.json()
                        st.success("Prédiction reçue ✅")

                        pred_class = data.get("prediction_class", None)
                        pred_label = data.get("prediction_label", None)

                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.metric("Classe", "—" if pred_class is None else str(pred_class))
                        with cc2:
                            st.metric("Label", "—" if pred_label is None else str(pred_label))

                        with st.expander("Voir la réponse JSON"):
                            st.json(data)

                except requests.exceptions.Timeout:
                    st.error("Timeout : l'API met trop de temps à répondre.")
                except Exception as e:
                    st.error(f"Erreur lors de l'appel API : {e}")

# -------------------------
# TAB 5 : RETRAIN + STATUS + RELOAD + MODEL INFO
# -------------------------
with tab5:
    st.subheader("Collecte + Réentrainement")
    st.caption("1) /collect → 2) /train → 3) suivre /train/status/{job_id} → reload modèle")

    # -------------------------
    # Infos modèle
    # -------------------------
    with st.expander("Infos modèle (API /model/info)"):
        try:
            mi = requests.get(MODEL_INFO_URL, headers=api_headers(), timeout=10)
            if mi.status_code >= 400:
                st.error(f"Erreur /model/info ({mi.status_code}): {mi.text}")
            else:
                st.json(mi.json())
        except Exception as e:
            st.error(f"Impossible d'appeler /model/info : {e}")

    st.divider()

    # -------------------------
    # (A) COLLECTE
    # -------------------------
    st.write("## 1) Collecte de données ")
    cA, cB, cC = st.columns(3)

    with cA:
        collect_rows = st.number_input("Nombre de lignes à collecter", min_value=1000, max_value=5_000_000, value=50_000, step=1000)
        mark_processed = st.checkbox("Marquer processed_for_model=1", value=True)

    with cB:
        # Path côté API (sur la machine/serveur où tourne l'API)
        collect_out_path = st.text_input("Chemin de sortie CSV", value="data/raw/collect_latest.csv")

    with cC:
        st.info(
            "le chemin est **sur le serveur de l'API**.\n\n"
            "Si dashboard et API sont sur la même machine, ok.\n"
            "Sinon, utilise un chemin que l'API peut écrire."
        )

    col_launch = st.button("Lancer", type="primary")

    if "last_collect_path" not in st.session_state:
        st.session_state["last_collect_path"] = ""

    if col_launch:
        collect_payload = {
            "n_rows": int(collect_rows),
            "out_path": collect_out_path,
            "mark_processed": bool(mark_processed),
        }
        with st.spinner("Collecte en cours (/collect)..."):
            try:
                r = requests.post(f"{API_BASE}/collect", json=collect_payload, headers=api_headers(), timeout=120)
                if r.status_code >= 400:
                    st.error(f"Erreur /collect ({r.status_code}) : {r.text}")
                else:
                    data = r.json()
                    st.success("Collecte terminée")
                    st.session_state["last_collect_path"] = data.get("out_path", collect_out_path)
                    with st.expander("Réponse /collect"):
                        st.json(data)
            except Exception as e:
                st.error(f"Erreur /collect : {e}")

    st.divider()

    # -------------------------
    # (B) TRAIN
    # -------------------------
    st.write("## 2) Réentrainement ")
    st.caption("Par défaut, on réutilise le CSV collecté ci-dessus si dispo.")

    with st.form("train_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            # 👉 data_path = chemin côté API
            data_path = st.text_input(
                "data_path (CSV côté API)",
                value=st.session_state.get("last_collect_path") or "data/raw/github_issues_full_500k.csv"
            )
            experiment = st.text_input("MLflow experiment", value="GitHub_Issues_XGBoost")
            n_samples = st.number_input("n_samples (max lignes utilisées)", min_value=1000, max_value=5_000_000, value=50_000, step=1000)

        with col2:
            random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42, step=1)
            tfidf_max_features = st.number_input("tfidf_max_features", min_value=500, max_value=20_000, value=1500, step=100)
            tfidf_ngram_max = st.number_input("tfidf_ngram_max", min_value=1, max_value=3, value=2, step=1)
            tfidf_stop_words = st.selectbox("tfidf_stop_words", options=["english", "none"], index=0)

        with col3:
            n_estimators = st.number_input("n_estimators", min_value=50, max_value=3000, value=200, step=50)
            learning_rate = st.number_input("learning_rate", min_value=0.01, max_value=1.0, value=0.08, step=0.01)
            max_depth = st.number_input("max_depth", min_value=2, max_value=20, value=6, step=1)
            subsample = st.slider("subsample", 0.1, 1.0, 0.9, 0.05)
            colsample_bytree = st.slider("colsample_bytree", 0.1, 1.0, 0.9, 0.05)

        launch_train = st.form_submit_button("Lancer /train")

    if "last_job_id" not in st.session_state:
        st.session_state["last_job_id"] = ""

    if launch_train:
        payload = {
            "data_path": data_path,
            "experiment": experiment,
            "n_samples": int(n_samples),
            "random_state": int(random_state),
            "tfidf_max_features": int(tfidf_max_features),
            "tfidf_ngram_max": int(tfidf_ngram_max),
            "tfidf_stop_words": tfidf_stop_words,
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "max_depth": int(max_depth),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "tree_method": "hist",
        }

        with st.spinner("Appel API /train ..."):
            try:
                r = requests.post(TRAIN_URL, json=payload, headers=api_headers(), timeout=30)
                if r.status_code >= 400:
                    st.error(f"Erreur /train ({r.status_code}) : {r.text}")
                else:
                    data = r.json()
                    st.success("Réentrainement déclenché")
                    st.session_state["last_job_id"] = data.get("job_id", "")
                    st.json(data)
            except Exception as e:
                st.error(f"Erreur /train : {e}")

    st.divider()

    # -------------------------
    # (C) SUIVI JOB + RELOAD
    # -------------------------
    st.write("## 3) Suivi du job")
    job_id = st.text_input("job_id (vide = dernier)", value=st.session_state.get("last_job_id", ""))
    auto_refresh = st.checkbox("Auto-refresh (2s)", value=False)

    if job_id:
        try:
            resp = requests.get(f"{STATUS_URL}/{job_id}", headers=api_headers(), timeout=10)
            if resp.status_code >= 400:
                st.error(f"Erreur status ({resp.status_code}) : {resp.text}")
            else:
                status = resp.json()
                st.write(f"**Status:** {status.get('status')}")
                if status.get("mlflow_run_id"):
                    st.write(f"**MLflow run_id:** `{status.get('mlflow_run_id')}`")

                with st.expander("Metrics"):
                    st.json(status.get("metrics"))

                with st.expander("stdout"):
                    st.code(status.get("stdout", ""), language="text")

                with st.expander("stderr"):
                    st.code(status.get("stderr", ""), language="text")

                if status.get("status") == "done":
                    if st.button("♻️ Reload modèle dans l'API (/reload-model)"):
                        rr = requests.post(RELOAD_URL, headers=api_headers(), timeout=10)
                        if rr.status_code >= 400:
                            st.error(f"Erreur reload ({rr.status_code}): {rr.text}")
                        else:
                            st.success("Modèle rechargé")
                            st.json(rr.json())

        except Exception as e:
            st.error(f"Erreur lecture status: {e}")

        if auto_refresh:
            time.sleep(2)
            st.rerun()
# =========================
# EXPORT CSV
# =========================
st.divider()
st.download_button(
    "Télécharger les données filtrées (CSV)",
    data=fdf.to_csv(
        index=False,
        quoting=csv.QUOTE_MINIMAL
    ).encode("utf-8"),
    file_name="github_issues_filtered.csv",
    mime="text/csv",
)