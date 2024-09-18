import gradio as gr
import PyPDF2
from transformers import AutoTokenizer, AutoModel
import torch
import weaviate
import cohere

auth_config = weaviate.AuthApiKey(api_key="**********************************")
client = weaviate.Client(
    url="https://wkoll9rds3orbu9fhzfr2a.c0.asia-southeast1.gcp.weaviate.cloud",
    auth_client_secret=auth_config
)
cohere_client = cohere.Client("**********************************")

def load_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

def upload_document_chunks(chunks):
    for idx, chunk in enumerate(chunks):
        embedding = get_embeddings(chunk)
        client.data_object.create(
            {"content": chunk},
            "Document",
            vector=embedding.tolist()
        )

def query_answer(query):
    query_embedding = get_embeddings(query)
    result = client.query.get("Document", ["content"])\
                .with_near_vector({"vector": query_embedding.tolist()})\
                .with_limit(3)\
                .do()
    return result

def generate_response(context, query):
    response = cohere_client.generate(
        model='command',
        prompt=f"Context: {context}\n\nQuestion: {query}?\nAnswer:",
        max_tokens=100
    )
    return response.generations[0].text.strip()

def qa_pipeline(pdf_file, query):
    document_text = load_pdf(pdf_file)
    document_chunks = [document_text[i:i+500] for i in range(0, len(document_text), 500)]

    upload_document_chunks(document_chunks)

    response = query_answer(query)
    context = ' '.join([doc['content'] for doc in response['data']['Get']['Document']])

    answer = generate_response(context, query)

    return context, answer

with gr.Blocks(theme="compact") as demo:
    gr.Markdown(
        """
        <div style="text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 20px; color: #2D3748;">
            📄 Interactive QA Bot 🔍
        </div>
        <p style="text-align: center; font-size: 16px; color: #4A5568;">
            Upload a PDF document, ask questions, and receive answers based on the document content.
        </p>
        <hr style="border: 1px solid #CBD5E0; margin: 20px 0;">
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="📁 Upload PDF", file_types=[".pdf"], show_label=True)
            query_input = gr.Textbox(
                label="❓ Ask a Question",
                placeholder="Enter your question here...",
                lines=1
            )
            submit_button = gr.Button("🔍 Submit")

        with gr.Column(scale=2):
            doc_segments_output = gr.Textbox(label="📜 Retrieved Document Segments", placeholder="Document segments will be displayed here...", lines=10)
            answer_output = gr.Textbox(label="💬 Answer", placeholder="The answer will appear here...", lines=3)

    submit_button.click(
        qa_pipeline,
        inputs=[pdf_input, query_input],
        outputs=[doc_segments_output, answer_output]
    )

    gr.Markdown(
        """
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
        """
    )

demo.launch()
