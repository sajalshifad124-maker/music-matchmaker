# Music Matchmaker

A local, content-based music recommender built from the Spotify audio-feature files in `data/`.

## Start it

1. Install the packages: `python -m pip install -r requirements.txt`
2. Run: `python -m streamlit run app.py`
3. Open the address shown in your browser.

Search for a title or artist, select a track, then explore tracks with a similar audio profile. The system standardises ten continuous audio features and ranks tracks using cosine similarity. It does not use listening history, so it works immediately without an account.

## Dataset

`data/data.csv` is the track-level source used by the app. The other provided files are preserved for future artist, genre, and trend views.
