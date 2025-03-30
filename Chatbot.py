# faculty_chatbot.py (Streamlit UI)

import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Model & Vectorizer filenames
MODEL_FILE = "faculty_vectorizer.pkl"
VECTORS_FILE = "faculty_vectors.pkl"
DATA_FILE = "college_faculty_details.csv"

# Load dataset and trained model
df = pd.read_csv(DATA_FILE)
with open(MODEL_FILE, "rb") as f:
    vectorizer = pickle.load(f)
with open(VECTORS_FILE, "rb") as f:
    faculty_vectors = pickle.load(f)

def get_faculty_info(query):
    """Find the most relevant faculty details based on user query."""
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, faculty_vectors).flatten()
    best_match_idx = similarities.argmax()
    if similarities[best_match_idx] > 0.1:  # Threshold to ensure relevance
        faculty = df.iloc[best_match_idx]
        if "who" in query.lower() or "teaches" in query.lower():
            return f"{faculty['Name']} teaches {faculty['Subjects']}."
        elif "contact" in query.lower() or "email" in query.lower():
            return f"You can contact {faculty['Name']} at {faculty['Email']}."
        elif "designation" in query.lower() or "position" in query.lower():
            return f"{faculty['Name']} is a {faculty['Designation']}."
        else:
            return f"{faculty['Name']} ({faculty['Designation']}) specializes in {faculty['Subjects']}. Contact: {faculty['Email']}."
    return "Sorry, I couldn't find relevant faculty information."

# Streamlit UI
st.title("Student - Faculty Chatbot")
st.write("Dataset based upon 3rd Year CSE Department")
st.write("Ask about faculty details (e.g., who teaches what, how to contact, etc.)")

# Example Queries
st.sidebar.title("💡 Example Queries")
st.sidebar.write("- Who teaches Machine Learning?")
st.sidebar.write("- What is Dr. Ashima Mehta's designation?")
st.sidebar.write("- How can I contact Ms. Sukrati Chaturvedi?")

# User Input
user_input = st.text_input("Type your question:")
if st.button("Ask Faculty Bot"):
    if user_input:
        response = get_faculty_info(user_input)
        st.success(response)
    else:
        st.warning("Please enter a question.")
