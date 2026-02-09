import pandas as pd
import mysql.connector
import joblib
import numpy as np
import mlflow
import mlflow.xgboost
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Correction encodage Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


DB_CONFIG = {
    'host': '92.113.22.16',
    'user': 'u616324536_big_data',
    'password': 'BigData2026@', 
    'database': 'u616324536_big_data',
    'port': 3306
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


print("[INFO] Chargement de 80 000 lignes (Version Light)...")
conn = get_connection()

query = """
SELECT title_text, body_text, language, stars, forks, num_comments, 
       num_labels, contains_bug, repo_age_days, created_hour, time_to_close
FROM github_issues_v1
WHERE state = 'CLOSED' 
  AND time_to_close > 0 
  AND time_to_close < 8000
LIMIT 80000 
"""
df = pd.read_sql(query, conn)
conn.close()
print(f"[OK] {len(df)} lignes chargées.")


def categorize_time_balanced(hours):
    if hours < 0.56: return 0      # Flash
    elif hours < 23.88: return 1   # Day
    else: return 2                 # Slow

df['target'] = df['time_to_close'].apply(categorize_time_balanced)


print("[INFO] Préparation des features...")
df['full_text'] = df['title_text'].fillna('') + " " + df['body_text'].fillna('')
df['text_len'] = df['full_text'].apply(len)


df['is_crash'] = df['full_text'].str.contains('crash|exception|error|fail|panic', case=False).astype(int)
df['is_feature'] = df['full_text'].str.contains('feature|add|request|support|implement', case=False).astype(int)

X = df[[
    'full_text', 'language', 'stars', 'forks', 
    'num_comments', 'num_labels', 'contains_bug', 
    'repo_age_days', 'created_hour', 'text_len',
    'is_crash', 'is_feature'
]]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("[INFO] Configuration du Pipeline XGBoost Light...")

preprocessor = ColumnTransformer(
    transformers=[
        
        ('txt', TfidfVectorizer(stop_words='english', max_features=1500), 'full_text'),
        
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['language']),
        
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['stars', 'forks', 'num_comments', 'num_labels', 'contains_bug', 
             'repo_age_days', 'created_hour', 'text_len', 'is_crash', 'is_feature'])
    ])


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,      
        learning_rate=0.1, 
        max_depth=6,           
        n_jobs=-1,           
        random_state=42,
        eval_metric='mlogloss',
        tree_method='hist'     
    ))
])


mlflow.set_experiment("GitHub_XGBoost_Light")

print("[INFO] Entraînement en cours (Optimisé)...")
with mlflow.start_run():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n[RESULT] Accuracy : {accuracy:.2%}")
    print("\n[RAPPORT DETAILLE]")
    target_names = ['Flash (<35m)', 'Day (<24h)', 'Slow (>24h)']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\n[MATRICE DE CONFUSION]")
    print(cm)
    
    mlflow.log_metric("accuracy", accuracy)
    joblib.dump(model, 'final_model_xgboost.pkl')

print("[SUCCESS] Modèle XGBoost Light sauvegardé.")