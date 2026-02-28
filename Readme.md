# MusicRecomAI — Music Genre Classification & Recommendation

A machine learning web app that classifies music into 10 genres using acoustic features, built on the GTZAN dataset.

## Features
- Genre classification using KNN and Random Forest
- Confidence score with secondary genre recommendation
- Demo mode (200 real test songs) + CSV upload mode
- Interactive genre probability distribution chart

## Models

| Model | Test Accuracy |
|-------|--------------|
| KNN (K=5) | 67.00% |
| Random Forest (100 trees) | 68.50% |

## Project Structure

    MusicRecomAI/
    ├── MusicRecAI.ipynb       # Full analysis notebook
    ├── app.py                 # Streamlit web app
    ├── features_30_sec.csv    # GTZAN dataset (pre-extracted)
    ├── knn_model.pkl          # Trained KNN model
    ├── rf_model.pkl           # Trained Random Forest model
    ├── scaler.pkl             # StandardScaler
    └── feature_columns.json   # Feature column order

## Run Locally

    pip install streamlit scikit-learn pandas numpy plotly
    streamlit run app.py

## Dataset

GTZAN Genre Collection — 1000 songs, 10 genres, 100 songs each.
Source: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
