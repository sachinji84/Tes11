import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

app = Flask(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_COR")

# Initialize the OpenAI client with LangChain
llm_model = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4")

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize ChromaDB
embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vector_db = Chroma(embedding_function=embedding_model,persist_directory="chromadb")

# Route to serve the HTML page
@app.route('/')
def index():
    """Serve the HTML page"""
    return send_from_directory('templates', 'upload.html')

# Route to upload a PDF document


@app.route('/upload', methods=['POST'])
def handle_upload():
    """Handle PDF document upload"""
    # file = request.files['document']
    file = request.files.get('document')  # Use .get() to avoid KeyError
    
    if file is None:
        return jsonify({"error": "No file part in the request"}), 400    

    # Check if the uploaded file is a PDF
    if file.filename.split('.')[-1].lower() != 'pdf':
        return jsonify({"error": "Only PDF files are allowed"}), 400

    # Save the file to the uploads directory
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract text from the PDF document using PyPDFLoader
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])

        # Chunk the content using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
            )
        chunks = text_splitter.split_text(content)

        # Create embeddings for each chunk and store them in ChromaDB
        vector_db.add_texts(
            texts=chunks, 
            ids=[f"{file.filename}-{i}" for i in range(len(chunks))])



        doc_id = len(documents) + 1
        documents[doc_id] = file.filename
        return jsonify({"message": "PDF document uploaded and processed successfully!", "doc_id": doc_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to interact with the document


@app.route('/interact', methods=['POST'])
def interact_with_document():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Perform similarity search in ChromaDB
        similar_docs = vector_db.similarity_search(user_query, k=5)
        relevant_content = "\n".join([doc.page_content for doc in similar_docs])

        # Define a prompt template
        template = (
            "You are a helpful assistant.\n"
            "Relevant document content:\n{relevant_content}\n"
            "User query: {user_query}"
        )

        # Create the prompt using LangChain's PromptTemplate
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["relevant_content", "user_query"]
        )

        # Create an LLMChain to use the prompt template and LLM
        chain = LLMChain(
            llm=llm_model,
            prompt=prompt_template
        )

        # Generate a response using the LLMChain
        response = chain.run(
            {"relevant_content": relevant_content, "user_query": user_query}
        )

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
