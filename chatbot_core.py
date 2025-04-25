
import json
import re
import random
import torch
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Preprocessing function to clean sentences
def preprocess_sentence(sentence):
    sentence = re.sub(r"[^\w\s]", "", sentence)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)

# Mean pooling function to get sentence-level embeddings
def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    return mean_pooled

# Function to generate BERT embeddings for a list of texts
def generate_bert_embeddings(texts, tokenizer, model):
    bert_embeddings = []
    for text in texts:
        preprocessed = preprocess_sentence(text)
        inputs = tokenizer(preprocessed, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        bert_embeddings.append(pooled)
    return torch.cat(bert_embeddings, dim=0)

# Load intents from JSON file
def load_intents(json_path="intents3.json"):
    with open(json_path, "r") as f:
        return json.load(f)

# Prepare data from intents: patterns and responses
def prepare_data(intents):
    patterns = []
    responses = []
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            responses.append(random.choice(intent["responses"]))
    return patterns, responses
