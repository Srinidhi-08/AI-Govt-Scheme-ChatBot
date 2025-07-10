import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA

app = Flask(__name__)
CORS(app)

# Load text documents
def load_documents_from_folder(folder_path, limit=5):
    docs = []
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    content = f.read()
                    docs.append(Document(page_content=content, metadata={"source": file}))
                count += 1
                if count >= limit:
                    break
        if count >= limit:
            break
    return docs

# Initialize models
llm = OllamaLLM(model="llama3")
embeddings = OllamaEmbeddings(model="llama3")


# Build or load FAISS index
FAISS_PATH = "faiss_index"
if os.path.exists(FAISS_PATH):
    vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    docs = load_documents_from_folder("schemes", limit=5)
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(FAISS_PATH)

retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@app.route("/")
def home():
    return "✅ Government Scheme Assistant running."

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        answer = qa_chain.run(query)
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)

