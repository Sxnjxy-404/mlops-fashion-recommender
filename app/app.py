import streamlit as st
import joblib

model, user_map, item_map, matrix = joblib.load("model/recommender.pkl")
inv_item_map = {v: k for k, v in item_map.items()}

st.title("🛍️ Personalized Fashion Recommendation")

user = st.text_input("Enter Customer ID")

if st.button("Recommend"):
    if user in user_map:
        uid = user_map[user]
        user_items = matrix[uid]
        recs = model.recommend(uid, user_items, N=5)

        items = []
        for r in recs:
            for v in r:
                if isinstance(v, (int,)) or str(v).isdigit():
                    items.append(inv_item_map[int(v)])
                    break

        st.success("Top recommendations:")
        for item in items:
            st.write(f"- Article ID: {item}")
    else:
        st.error("Customer ID not found")
