"""A small content-based music recommendation app for the supplied Spotify dataset."""

from __future__ import annotations

from pathlib import Path
import ast

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).parent
DATA_CANDIDATES = [
    PROJECT_DIR / "data" / "data.csv.gz",
    PROJECT_DIR / "data.csv.gz",
]
FEATURES = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]


def artists_as_text(value: object) -> str:
    """Turn the dataset's Python-list artist field into a readable string."""
    try:
        artists = ast.literal_eval(str(value))
        if isinstance(artists, list):
            return ", ".join(map(str, artists))
    except (ValueError, SyntaxError):
        pass
    return str(value)


@st.cache_data(show_spinner="Loading your music library…")
def load_music() -> pd.DataFrame:
    data_path = next((path for path in DATA_CANDIDATES if path.exists()), None)
    if data_path is None:
        expected = " or ".join(str(path.relative_to(PROJECT_DIR)) for path in DATA_CANDIDATES)
        raise FileNotFoundError(f"Dataset not found. Upload {expected} to the GitHub repository.")

    music = pd.read_csv(data_path)
    music = music.dropna(subset=["name", "artists", *FEATURES]).copy()
    music["artist_display"] = music["artists"].map(artists_as_text)
    music["label"] = music["name"] + " — " + music["artist_display"]
    # The dataset includes a small number of duplicated Spotify IDs.
    return music.drop_duplicates(subset="id").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def prepared_features(music: pd.DataFrame) -> np.ndarray:
    values = music[FEATURES].astype(float).to_numpy()
    # Standardising makes duration and loudness comparable with the 0–1 features.
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1
    return (values - mean) / std


def recommend(
    music: pd.DataFrame,
    scaled_features: np.ndarray,
    selected_index: int,
    count: int,
    same_era: bool,
    hide_explicit: bool,
) -> pd.DataFrame:
    """Return nearest tracks using cosine similarity on audio features."""
    query = scaled_features[selected_index]
    norms = np.linalg.norm(scaled_features, axis=1) * np.linalg.norm(query)
    scores = np.divide(scaled_features @ query, norms, out=np.zeros_like(norms), where=norms != 0)

    eligible = np.ones(len(music), dtype=bool)
    eligible[selected_index] = False
    if same_era:
        year = int(music.iloc[selected_index]["year"])
        eligible &= music["year"].between(year - 5, year + 5).to_numpy()
    if hide_explicit:
        eligible &= music["explicit"].eq(0).to_numpy()

    candidates = np.flatnonzero(eligible)
    ranked = candidates[np.argsort(scores[candidates])[::-1]][:count]
    result = music.iloc[ranked][["name", "artist_display", "year", "popularity", "explicit", "id"]].copy()
    result.insert(0, "match", scores[ranked])
    result["match"] = (result["match"] * 100).round().astype(int)
    return result


st.set_page_config(page_title="Music Matchmaker", page_icon="🎵", layout="wide")
st.title("🎵 Music Matchmaker")
st.caption("Discover songs with a similar sonic profile, using the supplied Spotify audio-feature dataset.")

try:
    music = load_music()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

features = prepared_features(music)

with st.sidebar:
    st.header("Recommendation settings")
    number = st.slider("Recommendations", min_value=5, max_value=25, value=10)
    same_era = st.checkbox("Keep within ±5 years", value=False)
    hide_explicit = st.checkbox("Hide explicit tracks", value=False)
    st.divider()
    st.caption(f"Searching {len(music):,} unique tracks")

query = st.text_input("Find a song or artist", placeholder="e.g. Blinding Lights or The Weeknd")
if not query.strip():
    st.info("Start by typing a song title or artist above.")
    st.stop()

matches = music[music["label"].str.contains(query, case=False, regex=False, na=False)].head(100)
if matches.empty:
    st.warning("No songs found. Try a shorter or different search.")
    st.stop()

chosen_label = st.selectbox("Choose a song", matches["label"].tolist())
selected_index = int(matches.index[matches["label"].eq(chosen_label)][0])
selected = music.iloc[selected_index]

st.subheader("Your starting point")
left, right = st.columns([3, 2])
with left:
    st.markdown(f"### {selected['name']}")
    st.write(selected["artist_display"])
    st.caption(f"Released {selected['year']} · Popularity {selected['popularity']}/100")
with right:
    st.metric("Energy", f"{selected['energy']:.0%}")
    st.metric("Danceability", f"{selected['danceability']:.0%}")

recommendations = recommend(music, features, selected_index, number, same_era, hide_explicit)
st.subheader("Recommended for you")
st.caption("Match reflects cosine similarity across energy, tempo, danceability, mood, acousticness, and related audio traits.")

table = recommendations.rename(
    columns={
        "match": "Match (%)",
        "name": "Song",
        "artist_display": "Artist",
        "year": "Year",
        "popularity": "Popularity",
        "explicit": "Explicit",
        "id": "Spotify ID",
    }
)
table["Explicit"] = table["Explicit"].map({0: "No", 1: "Yes"})
st.dataframe(table, use_container_width=True, hide_index=True)

csv = table.to_csv(index=False).encode("utf-8")
st.download_button("Download these recommendations", csv, "music-recommendations.csv", "text/csv")
