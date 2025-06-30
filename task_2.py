import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🤖 FAQ Chatbot")

faq_pairs = {
    "What is AI?": "AI stands for Artificial Intelligence.",
    "What is Python?": "Python is a programming language.",
    "What is Machine Learning?": "Machine Learning is a subset of AI.",
    "How does Streamlit work?": "Streamlit helps you build data apps easily."
}

questions = list(faq_pairs.keys())
answers = list(faq_pairs.values())

tfidf = TfidfVectorizer().fit_transform(questions)

user_q = st.text_input("Ask a quesstion:")
if user_q:
    user_vec = TfidfVectorizer().fit(questions + [user_q])
    vecs = user_vec.fit_transform(questions + [user_q])
    similarity = cosine_similarity(vecs[-1], vecs[:-1])
    best_match = questions[similarity.argmax()]
    st.success(f"Answer: {faq_pairs[best_match]}")
