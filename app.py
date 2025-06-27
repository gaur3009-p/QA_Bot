import os
import gradio as gr
import PyPDF2
import torch
import weaviate
from transformers import AutoTokenizer, AutoModel
from weaviate.classes.init import Auth
import cohere

# --- Configuration ---
WEAVIATE_URL = "*********************************************"
WEAVIATE_API_KEY = "*****************************************************************************************"
COHERE_API_KEY = "**********************************************"

# --- Initialize Clients ---
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    headers={"X-Cohere-Api-Key": COHERE_API_KEY}
)
cohere_client = cohere.Client(COHERE_API_KEY)

# --- Load Sentence Transformer ---
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# --- Utility Functions ---
def load_pdf(file):
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(file)
    return ''.join([page.extract_text() for page in reader.pages if page.extract_text()])

def get_embeddings(text):
    """Compute mean-pooled embeddings using a transformer."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

def upload_document_chunks(chunks):
    """Insert document chunks into Weaviate."""
    try:
        doc_collection = client.collections.get("Document")
    except Exception as e:
        raise RuntimeError("❌ Collection 'Document' not found. Make sure it's defined in your Weaviate schema.") from e

    for chunk in chunks:
        try:
            embedding = get_embeddings(chunk)
            doc_collection.data.insert(
                properties={"content": chunk},
                vector=embedding.tolist()
            )
        except Exception as e:
            print(f"⚠️ Skipped chunk due to error: {e}")

def query_answer(query):
    """Query Weaviate for top relevant document chunks."""
    query_embedding = get_embeddings(query)
    try:
        results = client.collections.get("Document").query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=3
        )
        return results.objects
    except Exception as e:
        print(f"⚠️ Query error: {e}")
        return []

def generate_response(context, query):
    """Generate a natural language response using Cohere."""
    response = cohere_client.generate(
        model='command',
        prompt=f"Context: {context}\n\nQuestion: {query}\nAnswer:",
        max_tokens=100
    )
    return response.generations[0].text.strip()

def qa_pipeline(pdf_file, query):
    """Main QA pipeline."""
    try:
        document_text = load_pdf(pdf_file)
        document_chunks = [document_text[i:i+500] for i in range(0, len(document_text), 500)]
        
        upload_document_chunks(document_chunks)
        top_docs = query_answer(query)
        context = ' '.join([doc.properties['content'] for doc in top_docs if 'content' in doc.properties])
        answer = generate_response(context, query)

        return str(context), str(answer)
    finally:
        client.close()

# --- Gradio UI ---
with gr.Blocks(theme="compact") as demo:
    gr.Markdown("""
        <div style="text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 20px; color: #2D3748;">
            📄 Interactive QA Bot 🔍
        </div>
        <p style="text-align: center; font-size: 16px; color: #4A5568;">
            Upload a PDF document, ask questions, and receive answers based on the document content.
        </p>
        <hr style="border: 1px solid #CBD5E0; margin: 20px 0;">
    """)

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="📁 Upload PDF", file_types=[".pdf"])
            query_input = gr.Textbox(label="❓ Ask a Question", placeholder="Enter your question here...")
            submit_button = gr.Button("🔍 Submit")

        with gr.Column(scale=2):
            doc_segments_output = gr.Textbox(label="📜 Retrieved Document Segments", lines=10)
            answer_output = gr.Textbox(label="💬 Answer", lines=3)

    submit_button.click(
        fn=qa_pipeline,
        inputs=[pdf_input, query_input],
        outputs=[doc_segments_output, answer_output]
    )

    gr.Markdown("""
        <style>
            body {
                background-color: #EDF2F7;
            }
            input[type="file"] {
                background-color: #3182CE;
                color: white;
                padding: 8px;
                border-radius: 5px;
            }
            button {
                background-color: #3182CE;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #2B6CB0;
            }
            textarea {
                border: 2px solid #CBD5E0;
                border-radius: 8px;
                padding: 10px;
                background-color: #FAFAFA;
            }
        </style>
    """)

demo.launch(share=True)
