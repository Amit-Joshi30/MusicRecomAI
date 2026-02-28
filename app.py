import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ── Load Models ───────────────────────────────────────────────
knn = pickle.load(open('knn_model.pkl', 'rb'))
rf = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)


# ── Recreate Test Set ─────────────────────────────────────────
df = pd.read_csv('features_30_sec.csv')
df.columns = df.columns.str.strip()
X = df.drop(['filename', 'length', 'label'], axis=1)
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test_scaled = scaler.transform(X_test)


# ── Genre Emoji Map ───────────────────────────────────────────
genre_emoji = {
    'blues': '🎸', 'classical': '🎻', 'country': '🤠',
    'disco': '🕺', 'hiphop': '🎤', 'jazz': '🎷',
    'metal': '🤘', 'pop': '🎙️', 'reggae': '🌴', 'rock': '🎵'
}


genre_desc = {
    'blues': 'Soulful, expressive music with deep emotional roots',
    'classical': 'Orchestral, harmonic and structurally complex music',
    'country': 'Acoustic, storytelling-driven music with rural roots',
    'disco': 'Upbeat, danceable music with strong rhythmic patterns',
    'hiphop': 'Rhythmic, bass-heavy music with vocal emphasis',
    'jazz': 'Improvisational, harmonically rich music',
    'metal': 'High energy, distorted and intense sonic experience',
    'pop': 'Catchy, melodic and widely accessible music',
    'reggae': 'Offbeat rhythms with Caribbean musical influences',
    'rock': 'Guitar-driven, energetic and diverse sonic range'
}


# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)


# ── Header ────────────────────────────────────────────────────
st.title("🎵 Music Genre Classification & Recommendation")
st.markdown("##### AI-powered genre prediction using acoustic features — built with KNN & Random Forest")
st.markdown("---")


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.radio("Select Model:", ["KNN (Primary)", "Random Forest (Comparison)"])
    model = knn if model_choice == "KNN (Primary)" else rf

    st.markdown("---")
    st.header("📊 Model Performance")
    if model_choice == "KNN (Primary)":
        st.metric("Test Accuracy", "67.00%")
        st.metric("Optimal K", "5")
        st.metric("CV Accuracy", "66.25%")
    else:
        st.metric("Test Accuracy", "68.50%")
        st.metric("No. of Trees", "100")
        st.metric("Best Genre", "Classical (F1: 0.86)")

    st.markdown("---")
    st.markdown("**Dataset:** GTZAN (1000 songs, 10 genres)")
    st.markdown("**Features:** 57 acoustic features")
    st.markdown("**Split:** 80% train / 20% test")


# ── Mode Tabs ─────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎧 Demo Mode — Test Songs", "📁 Upload Mode — Your CSV"])


# ════════════════════════════════════════════════════════════════
# TAB 1: DEMO MODE
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🎧 Select a Song from the Test Set")
    st.markdown("200 real songs held out during training — model has never seen these.")

    # Genre filter
    genre_filter = st.selectbox(
        "Filter by Actual Genre (optional):",
        ["All Genres"] + sorted(y_test.unique().tolist())
    )

    # Filter indices
    if genre_filter == "All Genres":
        filtered_indices = list(range(len(y_test)))
    else:
        filtered_indices = [i for i in range(len(y_test)) if y_test.iloc[i] == genre_filter]

    song_labels = [
        f"Song {i+1}  |  Actual: {y_test.iloc[i].title()}  {genre_emoji.get(y_test.iloc[i], '🎵')}"
        for i in filtered_indices
    ]

    selected = st.selectbox("Choose a song:", song_labels)
    song_index = filtered_indices[song_labels.index(selected)]

    if st.button("🎯 Predict Genre", key="demo_predict"):
        input_scaled = X_test_scaled[song_index].reshape(1, -1)
        predicted_genre = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        sorted_indices = np.argsort(probabilities)[::-1]

        primary_conf = probabilities[sorted_indices[0]] * 100
        second_genre = model.classes_[sorted_indices[1]]
        second_conf = probabilities[sorted_indices[1]] * 100
        actual_genre = y_test.iloc[song_index]

        st.markdown("---")

        # Correct/Wrong banner
        if predicted_genre == actual_genre:
            st.success(f"✅ Correct Prediction! The model got it right.")
        else:
            st.error(f"❌ Incorrect — Actual genre was **{actual_genre.title()}** {genre_emoji.get(actual_genre, '')}")

        # Main results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🎯 Predicted Genre")
            st.markdown(f"## {genre_emoji.get(predicted_genre, '🎵')} {predicted_genre.title()}")
            st.caption(genre_desc.get(predicted_genre, ''))

        with col2:
            st.markdown("### 📊 Confidence")
            st.markdown(f"## {primary_conf:.1f}%")
            st.progress(int(primary_conf))

        with col3:
            if primary_conf == 100.0:
                st.markdown("### 🎯 Unanimous Prediction")
                st.caption("All 5 neighbors agreed — maximum confidence")
            else:
                st.markdown("### 💡 You Might Also Like")
                st.markdown(f"## {genre_emoji.get(second_genre, '🎵')} {second_genre.title()}")
                st.caption(f"{second_conf:.1f}% secondary confidence")

        # Probability distribution
        st.markdown("---")
        st.subheader("📊 Full Genre Probability Distribution")
        prob_df = pd.DataFrame({
            'Genre': [f"{genre_emoji.get(g,'')} {g.title()}" for g in model.classes_],
            'Probability %': np.round(probabilities * 100, 2)
        }).sort_values('Probability %', ascending=False)

        st.bar_chart(prob_df.set_index('Genre'))

        # Feature snapshot
        st.markdown("---")
        st.subheader("🔬 Song's Acoustic Snapshot")
        raw_features = X_test.iloc[song_index]
        snap_cols = ['chroma_stft_mean', 'rms_mean', 'spectral_centroid_mean',
                     'zero_crossing_rate_mean', 'tempo',
                     'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean']
        snap_df = pd.DataFrame({
            'Feature': snap_cols,
            'Value': [round(raw_features[c], 4) for c in snap_cols]
        })
        st.dataframe(snap_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# TAB 2: UPLOAD MODE
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📁 Upload Your Own Song Features")
    st.markdown("Upload a CSV with pre-extracted GTZAN-format acoustic features.")

    with st.expander("ℹ️ What format should my CSV be in?"):
        st.markdown(f"""
        Your CSV should have these **{len(feature_columns)} columns** in order:
        """)
        st.code(", ".join(feature_columns))
        st.markdown("You can extract features from any audio file using `librosa` in Python.")

    uploaded_file = st.file_uploader("Upload CSV file:", type=['csv'])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)

        for col in ['filename', 'length', 'label']:
            if col in user_df.columns:
                user_df = user_df.drop(col, axis=1)

        if list(user_df.columns) == feature_columns:
            st.success(f"✅ Valid file! {len(user_df)} song(s) detected.")
            input_scaled = scaler.transform(user_df.values)

            if st.button("🎯 Predict Genre", key="upload_predict"):
                for idx in range(len(user_df)):
                    row_scaled = input_scaled[idx].reshape(1, -1)
                    predicted_genre = model.predict(row_scaled)[0]
                    probabilities = model.predict_proba(row_scaled)[0]
                    sorted_indices = np.argsort(probabilities)[::-1]
                    primary_conf = probabilities[sorted_indices[0]] * 100
                    second_genre = model.classes_[sorted_indices[1]]
                    second_conf = probabilities[sorted_indices[1]] * 100

                    st.markdown(f"### Song {idx+1}")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Predicted Genre",
                                  f"{genre_emoji.get(predicted_genre,'')} {predicted_genre.title()}")
                    with c2:
                        st.metric("Confidence", f"{primary_conf:.1f}%")
                    with c3:
                        if primary_conf == 100.0:
                            st.metric("Unanimous Prediction", "All 5 neighbors agreed")
                        else:
                            st.metric("You Might Also Like",
                                      f"{genre_emoji.get(second_genre,'')} {second_genre.title()} ({second_conf:.1f}%)")
                    st.markdown("---")
        else:
            st.error("⚠️ Column mismatch! Your CSV columns don't match the expected feature format.")
            st.markdown("**Expected columns:**")
            st.code(", ".join(feature_columns))
