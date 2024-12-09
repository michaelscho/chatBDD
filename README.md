# **ChatBDD**

This repository implements a Retrieval-Augmented Generation (RAG) pipeline for querying Latin texts using ChromaDB for document retrieval and LLaMA3.1. for natural language generation. The application includes a Gradio-based web interface for interactive querying LLaMA responses and dislplay document sources.

---

## **Features**

- **Latin-Specific Retrieval**: Uses `silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin` (https://huggingface.co/silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin) embeddings for Latin text processing in ChromaDB.
- **LLaMA 3.1 Integration**: Generates detailed responses using Ollama (required).
- **Source Transparency**: Displays relevant document excerpts and metadata alongside the generated responses.
- **Interactive Web Interface**: Provides a user-friendly Gradio interface for interaction.

---

## **Setup and Installation**

### **Prerequisites**

- Python 3.8 or higher
- pip
- Virtual environment (recommended)
- Ollama installed for LLaMA 3.1 ([installation guide](https://ollama.com/))

### **Steps**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/michaelscho/chatBDD.git
   cd chatBDD
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Ollama**:
   - Install Ollama:
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh  # For macOS/Linux
     ```
     For more details, refer to [Ollama's installation guide](https://ollama.com/).
   - Download the LLaMA 3.1 model:
     ```bash
     ollama pull llama3.1
     ```
   - Serve the model locally:
     ```bash
     ollama serve
     ```

5. **Run the Application**:
   Launch the Gradio interface:
   ```bash
   python chat_with_llama.py
   ```

6. **Access the Interface**:
   Open the URL provided by Gradio (e.g., `http://127.0.0.1:7860`) in your browser.

---

## **Usage**

### **1. Query the System**
- Type a query in the Gradio interface.
- The system will:
  - Retrieve relevant documents from ChromaDB.
  - Generate a response using LLaMA based on the retrieved context.
  - Display the response along with document excerpts and metadata.

### **2. Example Workflow**

#### **Query**:
`What are the rules for Lent fasting?`

#### **response**:
{'response': "Based on the provided text, which appears to be a portion of an ancient Christian canon law or decree, I can infer the following three important rules for Lent fasting:\n\n1. No arbitrary fasting: The text warns against imposing one-day fasts on oneself under the pretext of religion without consulting with one's bishop or their representative.\n2. Observance in congregation: It is prescribed to observe Lenten fasts together with other Christians, implying a communal aspect to the practice.\n3. Respect for church authorities: The text emphasizes that no one should impose fasting on themselves outside of the guidance and permission from their bishop or his delegate.\n\nThese rules are likely intended to ensure that Lenten fasting is observed in a way that maintains spiritual discipline while avoiding excessive zeal or individualistic practices 
that might be seen as contrary to the teachings of the Church.", 'sources': [{'id': 'edition-13-con-004', 'text': 'Siquis indictum ieiunium superbiendo contempserit et observare cum ceteris christianis noluerit in Gangrensi concilio praecipitur ut anathematizetur'}, {'id': 'edition-13-con-016', 'text': 'Presbyteri cum sacras festivitates populo annuntiant etiam ieiunium vigiliarum eos omnimodis servare moneant'}, {'id': 'edition-13-con-027', 'text': 'Ut nemo nisi consentiente proprio episcopo aut eius misso ieiunium sub obtentu religionis sibi imponat unum diem prae aliis excipiendo omnimodo interdicimus quod et factum displicet et in futurum fieri prohibemus quia plus causa hariolandi esse dinoscitur quam supplementum catholicae legis'}]}

---

## **Development Workflow**

### **Generate Embeddings for New Data**
- Add new TEI-encoded Latin texts to your dataset (e.g., `14.json`).
- Modify the script to process and embed the new data into ChromaDB.

### **Enhance LLaMA Responses**
- Fine-tune the prompt template in the `template` variable within `chat_with_llama.py`.

---

## **Contributing**

We welcome contributions! Feel free to fork this repository and submit pull requests for any enhancements or bug fixes.

---

## **License**

CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode.en)

---

## **Acknowledgments**

ChatBot is based on data provided by https://www.burchards-dekret-digital.de.

---
