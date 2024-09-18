# QA_Bot
# Workflow
#📄 Interactive QA Bot 🔍
This is an Interactive Question Answering Bot that allows users to upload a PDF document and ask questions related to its content. The bot uses Weaviate for document retrieval and Cohere for generating answers. The frontend is designed using Gradio, providing a user-friendly and visually appealing interface.

#🔧 Features
Upload PDF Documents: Users can upload a PDF file, and the document's content is processed and stored in Weaviate for embedding-based search.
Ask Questions: Users can ask questions about the uploaded document, and the bot will retrieve the most relevant document segments using embeddings and vector search.
Generate Answers: Using Cohere's language generation API, the bot generates coherent answers based on the retrieved content.
Modern UI: The interface is built with Gradio, featuring a simple and clean layout with a modern aesthetic.


#🛠️ Tech Stack
1.Gradio: For creating a clean and interactive web UI.
2.Weaviate: A vector database for storing and retrieving document embeddings.
3.Cohere: API for generating natural language responses.
4.Transformers: For text embeddings using sentence-transformers/all-MiniLM-L6-v2.
5.PyPDF2: For extracting text from uploaded PDF files.
6.Torch: For processing embeddings.

#🚀 How It Works
Upload a PDF: The user uploads a PDF document. The document is read, and the text is split into manageable chunks (500 characters each).
Generate Embeddings: Each chunk of text is converted into an embedding using a pre-trained transformer model (sentence-transformers/all-MiniLM-L6-v2).
Upload Chunks to Weaviate: The text chunks and their embeddings are stored in Weaviate to enable efficient vector-based search.
Query Processing: The user inputs a query, which is also converted into an embedding. Weaviate is queried to retrieve the document segments most similar to the query.
Answer Generation: The retrieved document segments are passed to Cohere's language generation API, which generates a detailed response based on the query and the context.
Display Results: The document segments and the generated answer are displayed on the interface for the user.
### 1. Clone the repository

```bash
git clone https://github.com/yourusername/interactive-qa-bot.git
cd interactive-qa-bot
```

###2. Install Dependencies
To install all the required dependencies, run the following:

```bash
Copy code
pip install -r requirements.txt
Or, manually install the dependencies if you don’t have a requirements.txt:
```
```bash
Copy code
pip install gradio transformers weaviate-client cohere pypdf2 torch
```
###3. Set Up Weaviate
Create a Weaviate instance on your preferred cloud provider (e.g., GCP, AWS).

Get the API Key and URL of your Weaviate instance.

Set up the schema in Weaviate if needed.

```python
Copy code
schema = {
    "class": "Document",
    "properties": [
        {
            "name": "content",
            "dataType": ["text"]
        }
    ],
    "vectorIndexType": "hnsw",
    "vectorizer": "none"
}

if not client.schema.exists("Document"):
    client.schema.create_class(schema)
```
###4. Set Up Cohere
Create a Cohere account at Cohere.
Get your API Key from the Cohere dashboard.
###5. Configure Environment Variables
You need to configure your Weaviate and Cohere API keys in the code. Replace the placeholders in app.py with your actual keys:
```python
Copy code
auth_config = weaviate.AuthApiKey(api_key="YOUR_WEAVIATE_API_KEY")
cohere_client = cohere.Client("YOUR_COHERE_API_KEY")
```

###6. Run the Application
To run the Gradio app locally, execute the following command:

```bash
Copy code
python app.py
After running the command, Gradio will launch a local server, and you’ll see a URL like http://localhost:7860. Open this URL in your browser to interact with the bot.
```
#🖼️ User Interface Overview
Upload PDF Section: Upload a PDF file by dragging and dropping or selecting a file from your computer.
Ask a Question: Enter a question related to the content of the uploaded document.
Retrieved Document Segments: The bot will show the most relevant segments from the document that are related to your query.
Generated Answer: A detailed answer generated using Cohere will be displayed.

#📦 Deployment
Deploying to Hugging Face Spaces
Create a Space on Hugging Face Spaces.
Set the SDK type to Gradio.
Add the necessary secrets for your Weaviate and Cohere API keys in the "Settings" section.
Push your repository to Hugging Face using Git:
```bash
Copy code
git add .
git commit -m "Initial commit"
git push origin main
```
Your app will be deployed, and Hugging Face will provide you with a public URL to share your QA bot with others.

#🛠️ Troubleshooting
Weaviate AuthApiKey Error: If you're facing issues related to Weaviate's AuthApiKey, ensure that you have the correct API key and are using the latest version of weaviate-client.

Performance Issues: If the embeddings take too long to process, consider optimizing the document chunk size or using a smaller model for embeddings.
