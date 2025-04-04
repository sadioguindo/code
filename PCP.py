#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Film Recommender", layout="wide")

# CSS personnalisé
st.markdown("""
<style>
.stApp {background-color: #87CEFA; font-family: 'Segoe UI';}
h1,h2,h3,h4 {color: white;}
.stButton>button {
    background-color: red;  /* Rouge vif */
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("user_ratings_genres_mov.csv")

data = load_data()

# Initialisation de la base d'utilisateurs simulée
if "users" not in st.session_state:
    st.session_state.users = {"admin": "admin"}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Menu principal
page = st.sidebar.radio("Navigation", ["Accueil", "Créer un compte", "Connexion", "Recommandations"])

# Accueil
if page == "Accueil":
    st.title("Bienvenue dans l'application de recommandation de films !")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Connexion"):
            st.session_state.page = "Connexion"
    with col2:
        if st.button("Créer un compte"):
            st.session_state.page = "Créer un compte"

# Création de compte
elif page == "Créer un compte":
    st.header("Créer un nouveau compte")
    new_user = st.text_input("Nom d'utilisateur", key="new_user")
    new_pwd = st.text_input("Mot de passe", type="password", key="new_pwd")
    confirm_pwd = st.text_input("Confirmer le mot de passe", type="password", key="confirm_pwd")

    if st.button("S'inscrire"):
        if new_user in st.session_state.users:
            st.warning("Ce nom d'utilisateur est déjà pris.")
        elif new_pwd != confirm_pwd:
            st.warning("Les mots de passe ne correspondent pas.")
        else:
            st.session_state.users[new_user] = new_pwd
            st.success("Compte créé avec succès ! Vous pouvez maintenant vous connecter.")

# Connexion
elif page == "Connexion":
    st.header("Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.success(f"Bienvenue, {username} !")
            st.session_state.logged_in = True
            st.session_state.page = "Recommandations"
        else:
            st.error("Identifiants incorrects.")

# Recommandations
elif page == "Recommandations":
    if not st.session_state.logged_in:
        st.warning("Veuillez vous connecter pour accéder aux recommandations.")
    else:
        st.header("Recommandation de Films")

        films = sorted(data['title'].unique())
        cols = st.columns(3)

        choices = []
        ratings = []
        for i, col in enumerate(cols):
            with col:
                choice = st.selectbox(f"Film {i+1}", films, key=f"film{i}")
                rating = st.slider("Note", 0.5, 5.0, 3.0, 0.5, key=f"rate{i}")
                choices.append(choice)
                ratings.append(rating)

        user_data = pd.DataFrame({
            'userId': ['new_user']*3,
            'title': choices,
            'rating': ratings,
            'genres': [data[data.title==f].genres.iloc[0] for f in choices]
        })

        st.write("### Vos films et genres sélectionnés :")
        st.dataframe(user_data[['title', 'rating', 'genres']])

        st.write("### Résumé des genres sélectionnés :")
        genres_summary = user_data['genres'].value_counts().reset_index()
        genres_summary.columns = ['Genres', 'Nombre de films']
        st.dataframe(genres_summary)

        methods = st.multiselect("Méthodes à appliquer", ["Item-User","User-Item","NMF","SVD","KNN","Contenu"])

        if st.button("Lancer recommandations"):
            data_updated = pd.concat([data,user_data])

            pivot_ut = data_updated.pivot_table(index='userId',columns='title',values='rating').fillna(0)
            pivot_tu = pivot_ut.T
            best_film = user_data.loc[user_data.rating.idxmax(),'title']

            if "Item-User" in methods:
                st.subheader("Item-User")
                sim = cosine_similarity(pivot_tu)
                sim_df = pd.DataFrame(sim, pivot_tu.index, pivot_tu.index)
                rec = sim_df[best_film].drop(best_film).sort_values(ascending=False).head(5)
                st.write(rec)

            if "User-Item" in methods:
                st.subheader("User-Item")
                sim = cosine_similarity(pivot_ut)
                sim_df = pd.DataFrame(sim, pivot_ut.index, pivot_ut.index)
                best_user = sim_df['new_user'].drop('new_user').idxmax()
                rec = pivot_ut.loc[best_user][pivot_ut.loc['new_user']==0].sort_values(ascending=False).head(5)
                st.write(rec)

            if "NMF" in methods:
                st.subheader("NMF")
                nmf = NMF(n_components=5)
                feats = nmf.fit_transform(pivot_tu)
                idx = pivot_tu.index.get_loc(best_film)
                sims = cosine_similarity([feats[idx]],feats)[0]
                top = pivot_tu.index[np.argsort(sims)[::-1][1:6]]
                st.write(top)

            if "SVD" in methods:
                st.subheader("SVD")
                svd = TruncatedSVD(n_components=5)
                feats = svd.fit_transform(pivot_tu)
                idx = pivot_tu.index.get_loc(best_film)
                sims = cosine_similarity([feats[idx]],feats)[0]
                top = pivot_tu.index[np.argsort(sims)[::-1][1:6]]
                st.write(top)

            if "KNN" in methods:
                st.subheader("KNN")
                knn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(pivot_tu)
                idx = pivot_tu.index.get_loc(best_film)
                _,indices = knn.kneighbors([pivot_tu.iloc[idx]])
                top = pivot_tu.index[indices[0][1:]]
                st.write(top)

            if "Contenu" in methods:
                st.subheader("Contenu")
                genres = list(set('|'.join(data_updated.genres).split('|')))
                for g in genres:
                    data_updated[g] = data_updated.genres.apply(lambda x:int(g in x.split('|')))
                film_vec = data_updated[data_updated.title==best_film][genres].values
                sims = cosine_similarity(film_vec, data_updated[genres])[0]
                top = data_updated.iloc[np.argsort(sims)[::-1][1:6]].title.unique()
                st.write(top)

        st.success("Recommandations terminées.")

