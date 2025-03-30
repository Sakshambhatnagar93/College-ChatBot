import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load faculty dataset
df = pd.read_csv("college_faculty_details.csv")

# Model & Vectorizer filenames
MODEL_FILE = "faculty_vectorizer.pkl"
VECTORS_FILE = "faculty_vectors.pkl"

# Train model and save it
vectorizer = TfidfVectorizer()
df['combined'] = df.apply(lambda x: f"{x['Name']} {x['Designation']} {x['Subjects']} {x['Email']}", axis=1)
faculty_vectors = vectorizer.fit_transform(df['combined'])

with open(MODEL_FILE, "wb") as f:
    pickle.dump(vectorizer, f)
with open(VECTORS_FILE, "wb") as f:
    pickle.dump(faculty_vectors, f)

print("Model trained and saved successfully!")
