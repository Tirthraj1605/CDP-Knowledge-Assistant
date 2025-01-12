import os
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import hdbscan
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
import streamlit as st

# Step 1: Scrape Data from Documentation Links
@st.cache_data
def scrape_documentation(link: str) -> List[str]:
    """
    Scrape documentation links and extract relevant text.
    """
    try:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, "html.parser")
        # Extracting all text from paragraphs (<p>) and headers (<h1>, <h2>, <h3>)
        texts = [tag.get_text(strip=True) for tag in soup.find_all(["p", "h1", "h2", "h3"])]
        return texts
    except Exception as e:
        st.error(f"Error scraping {link}: {e}")
        return []

@st.cache_data
def prepare_data():
    """
    Scrape, embed, and cluster the data for use in the assistant.
    """
    # Links to scrape
    documentation_links = {
        "Segment": "https://segment.com/docs/?ref=nav",
        "mParticle": "https://docs.mparticle.com/",
        "Lytics": "https://docs.lytics.com/",
        "Zeotap": "https://docs.zeotap.com/home/en-us/"
    }

    # Scrape and store all data
    all_data = []
    for name, link in documentation_links.items():
        texts = scrape_documentation(link)
        for text in texts:
            all_data.append({"source": name, "text": text})

    # Save scraped data to a DataFrame
    scraped_data = pd.DataFrame(all_data)

    # Generate embeddings
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    scraped_data["embedding"] = scraped_data["text"].apply(lambda x: bert_model.encode(x))

    # Clustering with HDBSCAN
    embeddings = list(scraped_data["embedding"])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean")
    scraped_data["cluster"] = clusterer.fit_predict(embeddings)

    # Calculate Cluster Centroids
    cluster_centroids = (
        scraped_data.groupby("cluster")["embedding"]
        .apply(lambda x: sum(x) / len(x))
        .to_dict()
    )

    # Fit NearestNeighbors for retrieval within clusters
    nn_models = {}
    for cluster_id in scraped_data["cluster"].unique():
        if cluster_id != -1:  # Exclude noise
            cluster_subset = scraped_data[scraped_data["cluster"] == cluster_id]
            nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
            nn.fit(list(cluster_subset["embedding"]))
            nn_models[cluster_id] = (nn, cluster_subset)

    return scraped_data, nn_models, bert_model, cluster_centroids

# Step 2: Load Model
@st.cache_resource
def load_qwen_model():
    """
    Load the Qwen model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    return tokenizer, model

# Step 3: Implement the Chatbot
class CDPKnowledgeAssistant:
    def __init__(self, data: pd.DataFrame, nn_models: Dict, bert_model, cluster_centroids, tokenizer, model):
        self.data = data
        self.nn_models = nn_models
        self.bert_model = bert_model
        self.cluster_centroids = cluster_centroids
        self.tokenizer = tokenizer
        self.model = model

    def retrieve_relevant_chunks(self, query: str) -> List[Dict]:
        """
        Retrieve relevant chunks using BERT embeddings and nearest neighbor search.
        """
        query_embedding = self.bert_model.encode(query)

        # Find the nearest cluster centroid
        cluster_distances = {
            cluster_id: sum((query_embedding - centroid) ** 2)
            for cluster_id, centroid in self.cluster_centroids.items()
        }
        best_cluster = min(cluster_distances, key=cluster_distances.get)

        # Handle the case where the best cluster is -1 (noise) silently
        if best_cluster == -1:
            # Perform global search without notifying the user
            nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
            nn.fit(list(self.data["embedding"]))
            distances, indices = nn.kneighbors([query_embedding])
            relevant_chunks = self.data.iloc[indices[0]].to_dict(orient="records")
            return relevant_chunks

        # Retrieve nearest neighbors within the best cluster
        nn, cluster_subset = self.nn_models[best_cluster]
        distances, indices = nn.kneighbors([query_embedding])
        relevant_chunks = cluster_subset.iloc[indices[0]].to_dict(orient="records")
        return relevant_chunks

    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate a response using the Qwen model, augmented with retrieved context.
        """
        if not context_chunks:
            return "I'm sorry, I couldn't find relevant information for your query."

        context_text = "\n".join([chunk["text"] for chunk in context_chunks])
        prompt = (
            f"Context:\n{context_text}\n\n"
            f"User Query:\n{query}\n\n"
            "Answer:"
        )
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(inputs.input_ids, max_length=200, num_return_sequences=1)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def answer_query(self, query: str) -> str:
        """
        Answer the user's query using RAG or external generation.
        """
        relevant_chunks = self.retrieve_relevant_chunks(query)
        return self.generate_response(query, relevant_chunks)

# Step 4: Streamlit Interface
def main():
    st.title("CDP Knowledge Assistant")
    st.write("Ask any question about Segment, mParticle, Lytics, or Zeotap!")

    # Prepare data and models
    scraped_data, nn_models, bert_model, cluster_centroids = prepare_data()
    tokenizer, model = load_qwen_model()

    assistant = CDPKnowledgeAssistant(scraped_data, nn_models, bert_model, cluster_centroids, tokenizer, model)

    # Input form
    query = st.text_input("Enter your query:", placeholder="How do I set up a new source in Segment?")
    if query:
        with st.spinner("Fetching response..."):
            response = assistant.answer_query(query)
        st.subheader("Response")
        st.write(response)

# Run Streamlit app
if __name__ == "__main__":
    main()
