"""Feature engineering for the Airbnb project.

add_basic_features() does the simple row-by-row transformations (parsing, dates, distances).
make_preprocessor() builds the encoders / TF-IDF. They are part of the sklearn pipeline,
so they are refitted on the training part of each cross-validation fold.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# last date in the dataset, used to turn dates into "number of days ago"
REFERENCE_DATE = pd.Timestamp("2017-10-05")

# approximate city centres (lat, lon)
CITY_CENTERS = {
    "NYC": (40.7580, -73.9855),      # Times Square
    "LA": (34.0522, -118.2437),      # Downtown
    "SF": (37.7880, -122.4075),      # Union Square
    "DC": (38.8977, -77.0365),       # White House
    "Chicago": (41.8827, -87.6233),  # The Loop
    "Boston": (42.3554, -71.0605),   # Boston Common
}

BOOLEAN_COLS = ["cleaning_fee", "host_has_profile_pic", "host_identity_verified", "instant_bookable"]

NUMERIC_FEATURES = [
    "accommodates", "bathrooms", "bedrooms", "beds", "latitude", "longitude",
    "number_of_reviews", "review_scores_rating", "host_response_rate", *BOOLEAN_COLS,
    "host_since_days", "last_review_days", "dist_to_center_km", "n_amenities", "description_len",
]
CATEGORICAL_FEATURES = ["property_type", "room_type", "bed_type", "cancellation_policy", "city", "neighbourhood"]


def parse_amenities(value):
    """'{TV,"Wireless Internet",Kitchen}' -> ['tv', 'wireless internet', 'kitchen']"""
    if not isinstance(value, str):
        return []
    items = value.strip("{}").replace('"', "").split(",")
    return [a.strip().lower() for a in items if a.strip() and "translation missing" not in a]


def distance_km(lat1, lon1, lat2, lon2):
    """Haversine distance between two points on Earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_basic_features(df):
    df = df.copy()

    for col in BOOLEAN_COLS:
        df[col] = df[col].map({"t": 1, "f": 0, True: 1, False: 0, "True": 1, "False": 0})

    # "95%" -> 95.0
    df["host_response_rate"] = pd.to_numeric(df["host_response_rate"].astype("string").str.rstrip("%"),
                                             errors="coerce")

    df["host_since_days"] = (REFERENCE_DATE - pd.to_datetime(df["host_since"], errors="coerce")).dt.days
    df["last_review_days"] = (REFERENCE_DATE - pd.to_datetime(df["last_review"], errors="coerce")).dt.days

    center_lat = df["city"].map({c: v[0] for c, v in CITY_CENTERS.items()})
    center_lon = df["city"].map({c: v[1] for c, v in CITY_CENTERS.items()})
    df["dist_to_center_km"] = distance_km(df["latitude"], df["longitude"], center_lat, center_lon)

    amenities = df["amenities"].apply(parse_amenities)
    df["n_amenities"] = amenities.str.len()
    df["amenities_text"] = amenities.str.join("|")

    df["description"] = df["description"].fillna("")
    df["description_len"] = df["description"].str.len()
    df["text"] = df["name"].fillna("") + " " + df["description"]

    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").fillna("unknown").astype(object)
    return df


def split_amenities(text):
    return text.split("|") if text else []


def make_preprocessor(model="tree"):
    """model="tree" for LightGBM, model="linear" for Ridge."""
    # one column per amenity (amenities present in at least 0.5% of listings)
    amenities = CountVectorizer(tokenizer=split_amenities, token_pattern=None, lowercase=False,
                                binary=True, min_df=0.005)
    # rare categories (< 20 listings) are grouped together
    categories = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True)

    if model == "tree":
        numeric = "passthrough"  # LightGBM handles missing values itself
        # the TF-IDF matrix is huge and sparse, so it is compressed to 50 dimensions for LightGBM
        text = make_pipeline(tfidf, TruncatedSVD(n_components=50, random_state=0))
    else:
        numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        text = tfidf  # Ridge works fine directly on the sparse matrix

    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categories, CATEGORICAL_FEATURES),
            ("amenities", amenities, "amenities_text"),
            ("text", text, "text"),
        ],
        sparse_threshold=1.0 if model == "linear" else 0.0,
    )
