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
from langchain.chains.combine_documents import create_stuff_documents_chain


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

# Initialize the vector database globally at the start
vectordb = None


def initialize_vector_db():
    global vectordb
    if vectordb is None:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
    return vectordb


# Initialize the vector store when the application starts
initialize_vector_db()


@app.route('/')
def index():
    """Serve the HTML page"""
    return send_from_directory('templates', 'upload.html')

# Handle PDF Upload and Process the Document


@app.route('/upload', methods=['POST'])
def handle_upload():
    """
    Handles the upload of a PDF document. 
    Extracts text from the document, splits it into chunks, embeds the chunks, and stores them in a vector database.

    Parameters:
    - request: The incoming Flask request object containing the uploaded file.

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
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH
        )

        # Use the global vector database
        vectordb.add_texts(
            texts=chunks,
            ids=[f"{file.filename}-{i}" for i in range(len(chunks))]
        )

        vectordb.persist()

        doc_id = len(documents) + 1
        documents[doc_id] = file.filename

        return jsonify({"message": "PDF document uploaded and processed successfully!", "doc_id": doc_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Interact with the Document and Retrieve Relevant Chunks
# Update the /interact/<int:doc_id> route to retrieve the most relevant chunks and generate a response:


@app.route('/interact/<int:doc_id>', methods=['POST'])
def interact_with_document(doc_id: int):
    """
    This function interacts with a document based on a user's query.
    It retrieves the relevant chunks of text from the document using a vector database,
    generates a response using a language model, and updates the conversation history.

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

    # Perform similarity search
    search_results = vectordb.similarity_search_with_score(user_query, k=5)
    relevant_content = "\n".join([result[0].page_content for result in search_results])

    # Retrieve conversation history or initialize it
    history = conversation_history.get(doc_id, "")
    
    # Convert the relevant content into a list of Document objects
    # context = [Document(page_content=relevant_content)]    

    # Define a prompt template
            # "Relevant document content:\n{relevant_content}\n"
    template = (
        "You are a helpful assistant.\n"
        "Conversation history:\n{history}\n"
        "Always Answer from given context. if you do not find the information say information is not available "
        "Context:\n{context}\n"
        "User query: {user_query}"
    )

    # Create the prompt using LangChain's PromptTemplate
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["history", "context", "user_query"]
    )

    # # Create an LLMChain to use the prompt template and LLM
    # chain = prompt_template | llm_model
    
    # Create the document chain
    
    
    document_chain = create_stuff_documents_chain(llm_model, prompt_template)    
    

    try:
        # Convert context to a string that can be passed into the template
        # context_str = "\n".join([doc.page_content for doc in context])
                
        response = document_chain.invoke({
            "history": history,
            "user_query": user_query,
            # "relevant_content": relevant_content,
            "context": [Document(page_content=relevant_content)]    

        })

        # Append the new query and response to the history
        conversation_history[doc_id] = history + f"User: {user_query}\nAssistant: {response.content}\n"

        return jsonify({"response": response.content})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


# Run the Flask Application
if __name__ == '__main__':
    app.run(debug=True)
