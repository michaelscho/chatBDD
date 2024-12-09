import torch
from transformers import AutoTokenizer, AutoModel
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# Load embedding model 
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

# Initialize LLaMA with Ollama (required)
llama = OllamaLLM(model="llama3.1")

# Define a prompt template
template = """
You are an expert assistant. Below is some context followed by a question. Use the context to generate the best possible answer.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Query and Generate Results
def query_and_generate_with_sources(query, k=3):
    # Retrieve relevant documents from ChromaDB
    results = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([result.page_content for result in results])
    sources = [{"id": result.metadata["id"], "text": result.page_content} for result in results]

    # Generate response with LLaMA
    chain = prompt | llama
    response = chain.invoke({"context": context, "question": query})

    return response, sources

# Gradio Interface
def chat_interface(query):
    response, sources = query_and_generate_with_sources(query, k=3)

    # Format sources for display
    source_texts = "\n\n".join([f"Source {idx + 1} (ID: {source['id']}): {source['text']}" for idx, source in enumerate(sources)])

    return response, source_texts

# Define Gradio Interface
interface = gr.Interface(
    fn=chat_interface,
    inputs="text",
    outputs=["text", "text"],
    title="ChatBDD",
    description="Ask questions and get answers with relevant source documents.",
    examples=[
        ["Rules for Lent fasting"],
        ["What are the principles of Lent fasting?"]
    ]
)

# Launch Gradio App (on http://127.0.0.1:7860)
if __name__ == "__main__":
    interface.launch()
