from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from dotenv import load_dotenv
import os
import PyPDF2

app = Flask(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_COR")

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# In-memory storage for simplicity (in production, use a database)
documents = {}


# Route to serve the HTML page
@app.route('/')
def index():
    return send_from_directory('templates', 'upload.html')

# Route to upload a PDF document
@app.route('/upload', methods=['POST'])
def upload_document():
    file = request.files['document']

    # Check if the uploaded file is a PDF
    if file.filename.split('.')[-1].lower() != 'pdf':
        return jsonify({"error": "Only PDF files are allowed"}), 400

    # Extract text from the PDF document
    pdf_reader = PyPDF2.PdfReader(file)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text()

    if not content:
        return jsonify({"error": "Failed to extract text from the PDF document"}), 500

    doc_id = len(documents) + 1
    documents[doc_id] = content
    return jsonify({"message": "PDF document uploaded and processed successfully!", "doc_id": doc_id})

# Route to interact with the document


@app.route('/interact/<int:doc_id>', methods=['POST'])
def interact_with_document(doc_id):
    if doc_id not in documents:
        return jsonify({"error": "Document not found"}), 404

    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    document_content = documents[doc_id]

    # Limit the document content to a maximum of 4000 tokens
    max_token_limit = 4000
    document_content = document_content[:max_token_limit]  # Adjust as needed

    # Create the messages for the chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",  "content": f"Document content (truncated):\n{document_content}"},
        {"role": "user", "content": user_query}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )

        # answer = response.choices[0].message["content"].strip()
        answer = response.choices[0].message.content
        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
