import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

udemy = pd.read_csv("udemy_courses.csv")
coursera = pd.read_csv("Coursera.csv")
coursea_data = pd.read_csv("coursea_data.csv")


# Build clean Udemy dataframe explicitly (NO rename / inplace issues)
udemy_clean = pd.DataFrame({
    "course_name": udemy["course_title"].astype(str),
    "difficulty_level": udemy["level"].astype(str),
    "skills": udemy["subject"].astype(str)
})

# Fix empty course names
udemy_clean.loc[
    udemy_clean["course_name"].str.strip() == "",
    "course_name"
] = np.nan

udemy_clean["course_name"] = udemy_clean["course_name"].fillna(
    pd.Series(
        "Udemy Course " + udemy_clean.index.astype(str),
        index=udemy_clean.index
    )
)

udemy_clean["course_rating"] = np.nan
udemy_clean["platform"] = "Udemy"

udemy_clean["content"] = (
    udemy_clean["course_name"] + " " +
    udemy_clean["skills"] + " " +
    udemy_clean["difficulty_level"]
)

udemy = udemy_clean

coursera = coursera.rename(columns={
    "Course Name": "course_name",
    "Difficulty Level": "difficulty_level",
    "Course Rating": "course_rating",
    "Skills": "skills"
})

coursera = coursera[
    ["course_name", "difficulty_level", "skills", "course_rating"]
].copy()

coursera["platform"] = "Coursera"

coursera["content"] = (
    coursera["course_name"].astype(str) + " " +
    coursera["skills"].astype(str) + " " +
    coursera["difficulty_level"].astype(str)
)

coursea_data = coursea_data.rename(columns={
    "course_title": "course_name",
    "course_difficulty": "difficulty_level",
    "course_rating": "course_rating"
})

coursea_data["skills"] = ""
coursea_data["platform"] = "Coursea_Data"

coursea_data = coursea_data[
    ["course_name", "difficulty_level", "skills", "course_rating"]
].copy()

coursea_data["content"] = (
    coursea_data["course_name"].astype(str) + " " +
    coursea_data["difficulty_level"].astype(str)
)

df = pd.concat(
    [udemy, coursera, coursea_data],
    ignore_index=True
)

df.drop_duplicates(
    subset=["course_name", "platform"],
    inplace=True
)

df["course_rating"] = pd.to_numeric(df["course_rating"], errors="coerce")
df["course_rating"].fillna(df["course_rating"].mean(), inplace=True)

df["rating_norm"] = df["course_rating"] / 5

df["difficulty_level"] = (
    df["difficulty_level"]
    .astype(str)
    .str.strip()
    .str.lower()
)

difficulty_map = {
    "beginner": "Beginner",
    "beginner level": "Beginner",
    "all levels": "Beginner",
    "all level": "Beginner",

    "intermediate": "Intermediate",
    "intermediate level": "Intermediate",
    "mixed": "Intermediate",
    "conversant": "Intermediate",

    "advanced": "Advanced",
    "expert": "Advanced",
    "expert level": "Advanced",

    "not calibrated": "Advanced"
}

df["difficulty_level_clean"] = df["difficulty_level"].map(difficulty_map)

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=6000
)

content_matrix = tfidf.fit_transform(df["content"])

def hybrid_recommend(skills, level, top_n=10):
    user_vec = tfidf.transform([skills])
    similarity = cosine_similarity(user_vec, content_matrix).flatten()

    df["score"] = (
        0.7 * similarity +
        0.3 * df["rating_norm"]
    )

    filtered = df[
        df["difficulty_level_clean"]
        .str.lower()
        .str.contains(level.lower(), na=False)
    ]

    result = filtered.sort_values(
        "score", ascending=False
    ).head(top_n)
    result["course_rating"] = result["course_rating"].round(1)

    return result.rename(columns={
        "course_name": "Course Name",
        "difficulty_level_clean": "Difficulty Level",
        "course_rating": "Course Rating"
    })[
        ["Course Name", "Difficulty Level", "Course Rating"]
    ]

