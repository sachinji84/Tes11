import os
import csv
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

app = Flask(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_COR")

# Initialize the OpenAI client with LangChain
llm_model = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4o")

# In-memory storage for simplicity (in production, use a database)
documents = {}

# In-memory storage for conversation history
conversation_history = {}

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def index():
    """Serve the HTML page"""
    return render_template('upload.html')

# Handle PDF Upload and Process the Document


@app.route('/upload', methods=['POST'])
def handle_upload():
    """
    Handles the upload of a PDF document. 
    Extracts text from the document, splits it into chunks, embeds the chunks, and stores them in a CSV file.

    Returns:
    - A JSON response indicating success or failure. If successful, it includes the document ID.
      If unsuccessful, it includes an error message.
    """
    file = request.files.get('document')  # Use .get() to avoid KeyError

    if file is None:
        return jsonify({"error": "No file part in the request"}), 400

    # Check if the uploaded file is a PDF
    if file.filename.split('.')[-1].lower() != 'pdf':
        return jsonify({"error": "Only PDF files are allowed"}), 400

    # Save the file to a temporary location
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract text from the PDF document using PyPDFLoader
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])

        # Initialize RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Split the document into chunks
        chunks = text_splitter.split_text(content)

        # Embed the chunks
        embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # CSV file to store chunk ID, chunk text, and embeddings
        csv_file_path = os.path.join(
            UPLOAD_FOLDER, f"{file.filename}_embeddings.csv")

        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["ChunkID", "ChunkText", "Embeddings"])

            for i, chunk in enumerate(chunks):
                embedding = embeddings_model.embed_query(chunk)
                writer.writerow([f"{file.filename}-{i}", chunk, embedding])

        doc_id = len(documents) + 1
        documents[doc_id] = file.filename

        return jsonify({"message": "PDF document uploaded and processed successfully!", "doc_id": doc_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Interact with the Document and Retrieve Relevant Chunks


@app.route('/interact/<int:doc_id>', methods=['POST'])
def interact_with_document(doc_id: int):
    """
    This function interacts with a document based on a user's query.
    It retrieves the relevant chunks of text from the CSV file using a similarity search
    and generates a response using a language model.

    Parameters:
    - doc_id (int): The unique identifier of the document.

    Returns:
    - dict: A JSON response containing either the generated response or an error message.
    """
    if doc_id not in documents:
        return jsonify({"error": "Document not found"}), 404

    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Load the CSV file containing embeddings
    csv_file_path = os.path.join(
        UPLOAD_FOLDER, f"{documents[doc_id]}_embeddings.csv")
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    try:
        # Embed the user's query
        query_embedding = embeddings_model.embed_query(user_query)

        # Read the CSV file and calculate similarities
        results = []
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                chunk_embedding = np.fromstring(row["Embeddings"][1:-1], sep=' ')
                similarity_score = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
                results.append((row["ChunkText"], similarity_score))

        # Sort results by similarity score in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        # Take the top 5 most relevant chunks
        relevant_content = "\n".join([result[0] for result in results[:5]])

        # Retrieve conversation history or initialize it
        history = conversation_history.get(doc_id, "")

        # Define a prompt template
        template = (
            "You are a helpful assistant.\n"
            "Conversation history:\n{history}\n"
            "Always Answer from given context. If you do not find the information, say the information is not available.\n"
            "Relevant document content:\n{relevant_content}\n"
            "User query: {user_query}"
        )

        # Create the prompt using LangChain's PromptTemplate
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["history", "relevant_content", "user_query"]
        )

        # Create an LLMChain to use the prompt template and LLM
        chain = prompt_template | llm_model

        response = chain.invoke({
            "history": history,
            "relevant_content": relevant_content,
            "user_query": user_query
        })

        # Append the new query and response to the history
        conversation_history[doc_id] = history + \
            f"User: {user_query}\nAssistant: {response.content}\n"

        return jsonify({"response": response.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask Application
if __name__ == '__main__':
    app.run(debug=True)
