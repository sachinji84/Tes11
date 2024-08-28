import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
# from langchain.runnables import RunnableSequence
from langchain.chains import LLMChain
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains.sequential import SimpleSequentialChain
# import tempfile

app = Flask(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_COR")

# Initialize the OpenAI client with LangChain
llm_model = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4")

# In-memory storage for simplicity (in production, use a database)
documents = {}

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route to serve the HTML page


@app.route('/')
def index():
    """Serve the HTML page"""
    return send_from_directory('templates', 'upload.html')

# Route to upload a PDF document


@app.route('/upload', methods=['POST'])
def handle_upload():
    """Handle PDF document upload"""
    file = request.files['document']

    # Check if the uploaded file is a PDF
    if file.filename.split('.')[-1].lower() != 'pdf':
        return jsonify({"error": "Only PDF files are allowed"}), 400

    # Save the file to a temporary location
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Extract text from the PDF document using PyPDFLoader
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])
        doc_id = len(documents) + 1
        documents[doc_id] = content
        return jsonify({"message": "PDF document uploaded and processed successfully!", "doc_id": doc_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # # Use LangChain's PyPDFLoader to load and extract text from the PDF
    # loader = PyPDFLoader(file)
    # docs = loader.load()
    # # Concatenate the text from all pages
    # content = "\n".join([doc.page_content for doc in docs])
    # if not content:
    #     return jsonify({"error": "Failed to extract text from the PDF document"}), 500
    # doc_id = len(documents) + 1
    # documents[doc_id] = content
    # return jsonify({"message": "PDF document uploaded and processed successfully!", "doc_id": doc_id})

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

    # Define a prompt template
    template = (
        "You are a helpful assistant.\n"
        "Document content (truncated):\n{document_content}\n"
        "User query: {user_query}"
    )

    # Create the prompt using LangChain's PromptTemplate
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["document_content", "user_query"]
    )

    # Create an LLMChain to use the prompt template and LLM
    # chain = SimpleSequentialChain(llm=llm_model,prompt_template=prompt_template)

    # Create an LLMChain to use the prompt template and LLM
    chain = LLMChain(
        llm=llm_model,
        prompt=prompt_template
    )

    # Create a RunnableSequence with the prompt template and LLM
    # sequence = RunnableSequence(prompt_template | llm_model)

    # # Define messages for the prompt
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": f"Document content (truncated):\n{document_content}\n{user_query}"}
    # ]

    # # Create the prompt using LangChain's ChatPromptTemplate
    # prompt_template = ChatPromptTemplate(messages=messages)

    # # prompt_template = ChatPromptTemplate(
    # #     system_prompt="You are a helpful assistant.",
    # #     user_prompt=f"Document content (truncated):\n{document_content}\n{user_query}"
    # # )

    # # Generate a response using the LangChain SimpleChain
    # chain = SimpleSequentialChain(llm=llm_model, prompt_template=prompt_template)

    try:
        # response = chain.run(input=user_query)
        # response = chain.run({"document_content": document_content, "user_query": user_query})
        response = chain.run(
            {"document_content": document_content,
             "user_query": user_query
             })

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
