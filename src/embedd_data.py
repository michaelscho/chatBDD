import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_chroma import Chroma

# 
# https://huggingface.co/silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin -> better retrieval!
# https://huggingface.co/mercelisw/xlm-roberta-base-extended-language-detection -> not tested
# xlm-roberta-large -> not good quality


# Load JSON data from book
path = os.path.join(os.getcwd(), "..", "data", "13.json")
print(f"Data path: {path}")

with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Load embedding model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Create custom embedding
class CustomEmbeddings:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def embed_documents(self, texts):
        """Embed multiple documents."""
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Convert numpy array to a Python list
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
        return embeddings

    def embed_query(self, text):
        """Embed a single query."""
        return self.embed_documents([text])[0]

custom_embeddings = CustomEmbeddings(tokenizer, model, device)

# Initialize ChromaDB with LangChain
vectorstore = Chroma(
    collection_name="BDD_2", # change collection name if necessary
    embedding_function=custom_embeddings,
    persist_directory="chroma_db" 
)

# Add Documents to database
for div in data["divs"]:
    text = div["text"]

    # Convert list metadata to JSON strings
    metadata = {
        "id": div["id"],
        "type": div["type"],
        "n": div["n"],
        "heading": div["heading"],
        "sameAs": div["metadata"]["sameAs"],
        "corresp": json.dumps(div["metadata"]["corresp"]),  # Convert list of corresponding ids in manuscripts to JSON string
    }

    # Add the document to the vectorstore
    vectorstore.add_texts(
        texts=[text],
        metadatas=[metadata],
        ids=[div["id"]]
    )


print(f"Database successfully created with {len(data['divs'])} documents.")

query = "Describe the rules for Lent fasting."

results = vectorstore.similarity_search(query, k=3)  # Retrieve top 3 matches, increase if necessary

for result in results:
    print(f"ID: {result.metadata['id']}")
    print(f"Heading: {result.metadata['heading']}")
    print(f"Correspondences: {json.loads(result.metadata['corresp'])}")
    print(f"Text: {result.page_content}\n")
