import pandas as pd
from pathlib import Path

# =====================
# PATHS
# =====================
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RAW_DIR / "github_issues_full.csv"  # ton CSV déjà exporté

# =====================
# TARGET
# =====================
def categorize_time_balanced(hours: float) -> int:
    if hours < 0.56:
        return 0  # Flash
    elif hours < 23.88:
        return 1  # Day
    else:
        return 2  # Slow

# =====================
# FEATURE ENGINEERING
# =====================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['full_text'] = df['title_text'].fillna('') + " " + df['body_text'].fillna('')
    df['text_len'] = df['full_text'].str.len()
    df['is_crash'] = df['full_text'].str.contains(
        'crash|exception|error|fail|panic', case=False
    ).astype(int)
    df['is_feature'] = df['full_text'].str.contains(
        'feature|add|request|support|implement', case=False
    ).astype(int)
    return df

# =====================
# PREPARE DATA
# =====================
def prepare_data(df: pd.DataFrame):
    X = df[[
        'full_text', 'language', 'stars', 'forks',
        'num_comments', 'num_labels', 'contains_bug',
        'repo_age_days', 'created_hour', 'text_len',
        'is_crash', 'is_feature'
    ]]
    y = df['target']
    return X, y

# =====================
# MAIN
# =====================
def main():
    # 1️⃣ Charger le CSV
    df = pd.read_csv(CSV_FILE)
    print(f"[OK] {len(df)} lignes chargées depuis CSV.")

    # 2️⃣ Target + features
    df['target'] = df['time_to_close'].apply(categorize_time_balanced)
    df = add_features(df)

    # 3️⃣ Préparer X et y
    X, y = prepare_data(df)

    # 4️⃣ Sauvegarde CSV transformé
    out_path = PROCESSED_DIR / "github_issues_full_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset complet sauvegardé → {out_path}")

    print("🎉 Toutes les données préparées avec succès !")
    return X, y, df

if __name__ == "__main__":
    X, y, df = main()
