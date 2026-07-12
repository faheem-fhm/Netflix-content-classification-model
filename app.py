import streamlit as st
import pickle
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Netflix Content Predictor", page_icon="🎬", layout="centered")

# ---------------------------------------------------------------
# Load model
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("🎬 Netflix Content Predictor")
st.write(
    "Fill in the details below. Text categories are converted to the encoded "
    "values the model was trained on."
)

st.warning(
    "These category lists and their encodings (alphabetical order, as scikit-learn's "
    "`LabelEncoder` does by default) are estimated since the original encoders weren't "
    "provided with the model. If you have the original training dataset, predictions "
    "can be made fully accurate by rebuilding the exact encoding used at training time."
)

st.divider()

# ---------------------------------------------------------------
# Category lists (sorted alphabetically -> index = encoded value)
# ---------------------------------------------------------------
COUNTRIES = sorted([
    "Argentina", "Australia", "Austria", "Belgium", "Brazil", "Canada", "Chile",
    "China", "Colombia", "Denmark", "Egypt", "France", "Germany", "Hong Kong",
    "India", "Indonesia", "Ireland", "Israel", "Italy", "Japan", "Lebanon",
    "Malaysia", "Mexico", "Netherlands", "Nigeria", "Norway", "Pakistan",
    "Philippines", "Poland", "Portugal", "Russia", "Singapore", "South Africa",
    "South Korea", "Spain", "Sweden", "Switzerland", "Taiwan", "Thailand",
    "Turkey", "United Arab Emirates", "United Kingdom", "United States", "Vietnam",
])

RATINGS = sorted([
    "G", "NC-17", "NR", "PG", "PG-13", "R", "TV-14", "TV-G", "TV-MA", "TV-PG",
    "TV-Y", "TV-Y7", "TV-Y7-FV", "UR",
])

GENRES = sorted([
    "Action & Adventure", "Anime Features", "Children & Family Movies", "Classic Movies",
    "Comedies", "Crime TV Shows", "Cult Movies", "Documentaries", "Dramas",
    "Horror Movies", "Independent Movies", "International Movies", "International TV Shows",
    "Kids' TV", "Music & Musicals", "Reality TV", "Romantic Movies", "Romantic TV Shows",
    "Sci-Fi & Fantasy", "Sports Movies", "Stand-Up Comedy", "TV Comedies", "TV Dramas",
    "Thrillers",
])

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

country_map = {name: i for i, name in enumerate(COUNTRIES)}
rating_map = {name: i for i, name in enumerate(RATINGS)}
genre_map = {name: i for i, name in enumerate(GENRES)}
month_map = {name: i + 1 for i, name in enumerate(MONTHS)}

# Output class 0/1 -> readable label.
# Edit this mapping if your target represents something else (e.g. Renew/Cancel, Hit/Flop).
OUTPUT_LABELS = {
    0: "Movie",
    1: "TV Show",
}

# ---------------------------------------------------------------
# Input form
# ---------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Input Features")

    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("Country", COUNTRIES)
        release_year = st.number_input("Release Year", min_value=1900, max_value=2100, value=2020, step=1)
        rating = st.number_input("Rating (encoded)", min_value=0, value=0, step=1)
        listed_in = st.selectbox("Genre (Listed In)", GENRES)

    with col2:
        year_added = st.number_input("Year Added to Netflix", min_value=1900, max_value=2100, value=2021, step=1)
        month_added = st.selectbox("Month Added", MONTHS)
        content_age = st.number_input("Content Age (years since release)", min_value=0, value=1, step=1)

    submitted = st.form_submit_button("Predict")

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame(
        [[
            country_map[country],
            release_year,
            rating,
            genre_map[listed_in],
            year_added,
            month_map[month_added],
            content_age,
        ]],
        columns=[
            "country",
            "release_year",
            "rating",
            "listed_in",
            "year_added",
            "month_added",
            "content_age",
        ],
    )

    st.write("### Encoded Input Sent to Model")
    st.dataframe(input_df, use_container_width=True)

    prediction = model.predict(input_df)[0]
    predicted_label = OUTPUT_LABELS.get(prediction, str(prediction))
    st.write("### Prediction Result")
    st.success(f"Predicted: **{predicted_label}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        proba_df = pd.DataFrame({
            "Class": [OUTPUT_LABELS.get(c, str(c)) for c in model.classes_],
            "Probability": proba,
        }).sort_values("Probability", ascending=False)
        st.write("### Class Probabilities")
        st.bar_chart(proba_df.set_index("Class"))
        st.dataframe(proba_df, use_container_width=True)

st.divider()
st.caption("Model: RandomForestClassifier | Built with Streamlit")