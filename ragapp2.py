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
from langchain.schema import Document

app = Flask(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_COR")
CHROMA_DB_PATH = "chromadb"  # Define the path where your ChromaDB will be stored

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
    return send_from_directory('templates', 'upload.html')

# Handle PDF Upload and Process the Document
# Update the /upload route to include chunking, embedding, and storing in ChromaDB:


@app.route('/upload', methods=['POST'])
def handle_upload():
    """Handle PDF document upload"""
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

        # chunks = []
        # for doc in docs:
        #     chunks.extend(text_splitter.split_text(doc.page_content))
        chunks = text_splitter.split_text(content)

        # Embed the chunks
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH
        )

        # for i, chunk in enumerate(chunks):
        #     vectordb.add_documents([Document(page_content=chunk)], metadata={"chunk_id": i, "file_name": file.filename})

        vectordb.add_texts(
            texts=chunks,
            ids=[f"{file.filename}-{i}" for i in range(len(chunks))])

        vectordb.persist()

        doc_id = len(documents) + 1
        documents[doc_id] = file.filename
        return jsonify({"message": "PDF document uploaded and processed successfully!", "doc_id": doc_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Interact with the Document and Retrieve Relevant Chunks
# Update the /interact/<int:doc_id> route to retrieve the most relevant chunks and generate a response:


@app.route('/interact/<int:doc_id>', methods=['POST'])
def interact_with_document(doc_id):
    if doc_id not in documents:
        return jsonify({"error": "Document not found"}), 404

    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    file_name = documents[doc_id]

    # Retrieve the embeddings from ChromaDB
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    # Create a retriever using as_retriever()
    # retriever = vectordb.as_retriever()

    # search_results = retriever.retrieve(user_query, k=5)
    # relevant_content = "\n".join([result.page_content for result in search_results])

    # Perform similarity search
    search_results = vectordb.similarity_search_with_score(user_query, k=5)
    relevant_content = "\n".join([result[0].page_content for result in search_results])
    
    # Retrieve conversation history or initialize it
    history = conversation_history.get(doc_id, "")

    # Define a prompt template
    template = (
        "You are a helpful assistant.\n"
        "Conversation history:\n{history}\n"
        "Always Answer from given context. if you do not find the information say information is not available "
        "Relevant document content:\n{relevant_content}\n"
        "User query: {user_query}"
    )

    # Create the prompt using LangChain's PromptTemplate
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["history", "relevant_content", "user_query"]
    )

    # Create an LLMChain to use the prompt template and LLM

    # chain = LLMChain(llm=llm_model, prompt=prompt_template)
    chain = prompt_template | llm_model

    try:
        response = chain.invoke({
            "history": history,
            "relevant_content": relevant_content,
            "user_query": user_query
        })
        
        # Append the new query and response to the history
        conversation_history[doc_id] = history + f"User: {user_query}\nAssistant: {response.content}\n"

        return jsonify({"response": response.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask Application
if __name__ == '__main__':
    app.run(debug=True)
