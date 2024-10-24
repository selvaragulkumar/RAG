import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Declare your API key directly in the code
gemini_api_key = "AIzaSyA4l7y413rovnEvgVJuU8OAey-MuU7QRx0"

# Configure the API key for Gemini
genai.configure(api_key=gemini_api_key)

# Streamlit app
st.title("RAG Chatbot with Gemini API")

# Function to ask Gemini a question and execute code
def ask_gemini(paragraph, question):
    model = genai.GenerativeModel(model_name='gemini-1.5-pro')
    
    # Combine paragraph with user question
    prompt = f"{paragraph}\n\nQuestion: {question}"
    
    # Generate content using the Gemini model
    response = model.generate_content(prompt, tools='code_execution')
    
    return response.text

# Function to extract relevant paragraph using TF-IDF
def get_relevant_paragraph(documents, query):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents + [query])
    
    # Compute cosine similarity between the query and all documents
    cosine_similarities = np.dot(tfidf_matrix[-1], tfidf_matrix[:-1].T).toarray()[0]
    
    # Get the index of the most relevant paragraph
    relevant_index = np.argmax(cosine_similarities)
    
    return documents[relevant_index]

# User input for document upload
uploaded_file = st.file_uploader("Upload a text document", type="txt")

# User input for chat
if uploaded_file is not None:
    # Read the uploaded document
    documents = uploaded_file.read().decode("utf-8").split('\n\n')  # Split paragraphs by two newlines

    # Ask user for a query
    user_query = st.text_input("Ask the RAG Chatbot:", "")

    if st.button("Send"):
        if user_query:
            # Get the most relevant paragraph
            relevant_paragraph = get_relevant_paragraph(documents, user_query)
            
            # Get the response from Gemini API
            response_text = ask_gemini(relevant_paragraph, user_query)
            
            st.write("RAG Chatbot's response:")
            st.write(response_text)

# Display instructions
st.write("### Instructions:")
st.write("1. Upload a text document containing paragraphs.")
st.write("2. Type a question related to the content of the document.")
st.write("3. Click 'Send' to get a response from the RAG Chatbot.")
