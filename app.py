import streamlit as st
from chatbot_core import preprocess_sentence, generate_bert_embeddings, prepare_data, load_intents
from sklearn.metrics.pairwise import cosine_similarity
import torch
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


# Load data
intents = load_intents('intents3.json')
patterns, responses = prepare_data(intents)

# Load BERT once
from transformers import BertTokenizer, BertModel

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, model = load_model()

# Prepare pattern embeddings
bert_embeddings_patterns = generate_bert_embeddings(patterns, tokenizer, model)

# Match and get best response
def get_best_response(user_input):
    cleaned_input = preprocess_sentence(user_input)
    user_embedding = generate_bert_embeddings([cleaned_input], tokenizer, model)
    similarity = cosine_similarity(user_embedding, bert_embeddings_patterns)
    best_idx = similarity.argmax()
    return responses[best_idx]

# Streamlit UI
st.title("🤖 ML Doubt Clarifier Bot")
st.markdown("Ask me anything about machine learning topics!")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:")

if st.button("Send") or user_input:
    if user_input.lower() == 'exit':
        st.session_state.chat_history.append(("SONI", "Goodbye! Have a great day!"))
    else:
        response = get_best_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("SONI", response))

# Show chat history
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.write(f"**You:** {msg}")
    else:
        st.success(f"**SONI:** {msg}")


