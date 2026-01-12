import pandas as pd
import scipy.sparse as sp
import implicit
import joblib
import os

DATA_PATH = "data/hm/transactions.csv"
MODEL_PATH = "model/recommender.pkl"

os.makedirs("model", exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df[['customer_id', 'article_id']]

print("Encoding users and items...")
user_map = {u:i for i,u in enumerate(df['customer_id'].unique())}
item_map = {i:j for j,i in enumerate(df['article_id'].unique())}

df['u'] = df['customer_id'].map(user_map)
df['i'] = df['article_id'].map(item_map)

print("Building sparse matrix...")
matrix = sp.coo_matrix(( [1]*len(df), (df['u'], df['i']) ))

print("Training model...")
model = implicit.als.AlternatingLeastSquares(factors=50)
model.fit(matrix)

print("Saving model...")
joblib.dump((model, user_map, item_map), MODEL_PATH)

print("✅ Training complete and model saved.")
