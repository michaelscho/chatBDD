import json
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Reload XML-RoBERTa and ChromaDB
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

class CustomEmbeddings:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def embed_documents(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

custom_embeddings = CustomEmbeddings(tokenizer, model, device)
vectorstore = Chroma(
    collection_name="BDD_2",
    embedding_function=custom_embeddings,
    persist_directory="chroma_db"
)

# Step 2: Initialize Ollama with LLaMA 3.1
llama = OllamaLLM(model="llama3.1")

# Step 3: Define a Prompt Template for LLaMA
template = """
You are an expert assistant. Below is some context followed by a question. Use the context to generate the best possible answer.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Step 4: Query the Database and Generate Results
def query_and_generate_with_sources(query, k=3):
    # Retrieve relevant documents from ChromaDB
    results = vectorstore.similarity_search(query, k=k)

    # Extract texts and IDs from the results
    context = "\n\n".join([result.page_content for result in results])
    sources = [{"id": result.metadata["id"], "text": result.page_content} for result in results]

    # Combine the context and query into the prompt
    chain = prompt | llama
    response = chain.invoke({"context": context, "question": query})

    # Return the response along with the source documents
    return {
        "response": response,
        "sources": sources
    }


# Step 5: Run the Pipeline
#query = "What are the three most important rules for Lent fasting?"
query = "List all passages concerning 'hariolandi'."

response = query_and_generate_with_sources(query, k=3)
print("Response:")
print(response)
