import os
import datetime
import logging
import requests
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.language_models import LLM

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv()





class OllamaLLM(LLM):
    model_name: str = "llama2"

    # for locat deployment
    #base_url: str = "http://localhost:11434"

    #when running via docker both in dev and prouction
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    #base_url: str = "http://host.docker.internal:11434"
    
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 25
    max_tokens: int = 200  # or any default limit you prefer
    logging.info(f"tempreture:{temperature}")

    def _call(self, prompt: str, stop=None) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "num_predict": self.max_tokens
                }
            }
        )
        response.raise_for_status()
        return response.json()["response"]

    @property
    def _llm_type(self) -> str:
        return "ollama-llama2"

# def load_model_and_tokenizer():
#     return OllamaLLM(), None  # tokenizer not needed with Ollama

def load_model_and_tokenizer(
    model_name="llama2",
    temperature=0.2,
    top_p=0.9,
    top_k=20,
    max_tokens=200
):
    llm = OllamaLLM(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens
    )
    return llm, None  # tokenizer not needed with Ollama


# Prompt templates
pdf_prompt_template = """You are a helpful assistant. Use the context to answer the question.
         If the answer isn't in the context, reply accordingly.
Context:
{context}

Question: {question}

Respond conversationally but informatively:"""
pdf_prompt = PromptTemplate(template=pdf_prompt_template, input_variables=["context", "question"])

template_facts = """You are a knowledgeable assistant who speaks naturally. Structure your response shortly :
Keep it short and to the point.

Provide core facts conversationally if any
Add interesting details if any
End with an inviting question

Question: {question}

Informative answer:"""
fact_prompt = PromptTemplate(template=template_facts, input_variables=["question"])
template_explain="""
You are a friendly, helpful chatbot.

If the input is a casual question or greeting (e.g., "how are you?"), respond naturally and briefly.

Otherwise, explain the following topic in simple, accurate, and to-the-point terms:

Topic: {question}

Respond in chatbot style:
"""


template_explain = """Explain the following in simple terms and to the point accurate information:
Keep it short and to the point.
Topic: {question}

Respond in chatbot style:"""
explain_prompt = PromptTemplate(template=template_explain, input_variables=["question"])

greeting_prompt_template = """You are a virtual assistant. Respond to greetings or casual inputs in a professional  way.

Guidelines:
- Keep it short and to the point.
- Include only one relevant emoji (if appropriate).
- Do NOT include any personal information
- Be context-aware (consider time of day).
- Answer the question asked (if any question)

Input: {query}
Current time: {time}
Response:"""



greeting_prompt = PromptTemplate(template=greeting_prompt_template, input_variables=["query", "time"])

# Load model (Ollama)
#llm, _ = load_model_and_tokenizer()
llm, _ = load_model_and_tokenizer(
    temperature=0.2,
    top_p=0.8,
    top_k=20,
    max_tokens=500
)


# Assign same LLM instance to all modes since Ollama doesn't expose sampling controls
llm_pdf = llm
llm_safe = llm
llm_creative = llm

# LLM Chains
pdf_chain = LLMChain(llm=llm_pdf, prompt=pdf_prompt)
fact_chain = LLMChain(llm=llm_creative, prompt=fact_prompt)
explain_chain = LLMChain(llm=llm_creative, prompt=explain_prompt)
greeting_chain = LLMChain(llm=llm_safe, prompt=greeting_prompt)

# Embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")



def create_vector_store(pdf_path: str, store_name: str = "vectorstore"):
    """Process PDF and merge with existing vector store if available"""
    
    # Load new PDF
    loader = PyPDFLoader(pdf_path)
    new_docs = loader.load()
    for doc in new_docs:
        doc.page_content = clean_text(doc.page_content)
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    new_chunks = splitter.split_documents(new_docs)
    
    # Create FAISS index for new content
    new_db = FAISS.from_documents(new_chunks, embedder)
    
    # Try loading existing index
    if os.path.exists(store_name):
        existing_db = FAISS.load_local(store_name, embedder, allow_dangerous_deserialization=True)
        existing_db.merge_from(new_db)
        vectordb = existing_db
    else:
        vectordb = new_db
    
    # Save merged index
    vectordb.save_local(store_name)
    return len(new_chunks)  # Return number of new chunks added

import shutil

def clear_vector_store(store_name: str = "vectorstore"):
    """Deletes the FAISS vector store and its hash file after user confirmation."""
    confirm = input(f"Are you sure you want to delete the vector store '{store_name}'? (y/n): ")
    if confirm.lower() == "y":
        if os.path.exists(store_name):
            shutil.rmtree(store_name)
            print(f"Vector store '{store_name}' deleted.")
        else:
            print(f"Vector store '{store_name}' not found.")

        hash_file = store_name + "_hashes.json"
        if os.path.exists(hash_file):
            os.remove(hash_file)
            print(f"Hash file '{hash_file}' deleted.")
        else:
            print(f"No hash file '{hash_file}' to delete.")
    else:
        print("Aborted: Vector store was not deleted.")

def create_vector_store2(pdf_path: str, store_name: str = "vectorstore"):
    """Process PDF and merge with existing vector store if available"""
    
    # Load new PDF
    loader = PyPDFLoader(pdf_path)
    new_docs = loader.load()
    for doc in new_docs:
        doc.page_content = clean_text(doc.page_content)
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    new_chunks = splitter.split_documents(new_docs)
    
    # Create FAISS index for new content
    new_db = FAISS.from_documents(new_chunks, embedder)
    
    # Try loading existing index
    if os.path.exists(store_name):
        existing_db = FAISS.load_local(store_name, embedder, allow_dangerous_deserialization=True)
        existing_db.merge_from(new_db)
        vectordb = existing_db
    else:
        vectordb = new_db
    
    # Save merged index
    vectordb.save_local(store_name)
    return len(new_chunks)  # Return number of new chunks added
# Vector store functions
# def create_vector_store(pdf_path: str):
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()
#     for doc in docs:
#         doc.page_content = clean_text(doc.page_content)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
#     chunks = splitter.split_documents(docs)
#     vectordb = FAISS.from_documents(chunks, embedding=embedder)
#     vectordb.save_local("vectorstore")


def clean_text(text: str) -> str:
    """
    Cleans PDF text by:
    1. Replacing newlines with spaces
    2. Removing bullet points (•, -, *, etc.)
    3. Collapsing multiple spaces into one
    4. Removing leading/trailing spaces
    """
    # Replace newlines and tabs
    text = re.sub(r'[\n\t]+', ' ', text)
    
    # Remove bullet points and special markers
    text = re.sub(r'[\•\-\*\▪\▫\‣\⁃]', ' ', text)
    
    # Collapse spaces and trim
    text = re.sub(r' +', ' ', text).strip()
    
    return text
def load_vector_store():
    vectordb = FAISS.load_local("vectorstore", embedder, allow_dangerous_deserialization=True)
    try:
        # Sample query just to log a chunk
        sample_result = vectordb.similarity_search("sample", k=1)
        if sample_result:
            logging.info("[VectorStore] Sample chunk:\n%s", sample_result[0].page_content[:300])
        else:
            logging.info("[VectorStore] No chunks found.")
    except Exception as e:
        logging.warning(f"[VectorStore] Could not log sample chunk: {e}")
    return vectordb

# def load_vector_store():
#     return FAISS.load_local("vectorstore", embedder, allow_dangerous_deserialization=True)

# Prompt fallback routing
def choose_fallback_prompt(query: str):
    if any(word in query.lower() for word in ["how", "why", "explain", "difference", "what is"]):
        logging.info("Selected prompt: explain_chain")
        return explain_chain
    logging.info("Selected prompt: fact_chain")
    return fact_chain

# Greeting check
def is_greeting_query(query: str) -> bool:
    query = query.lower().strip()
    greetings = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"}
    return any(greet in query for greet in greetings)

# Main answer logic
import re

def clean_text(text: str) -> str:
    # Replace newlines with space
    text = text.replace('\n', ' ')
    # Remove bullet points (• or other similar chars)
    text = re.sub(r'[•◦▪]', '', text)
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    # Trim leading/trailing whitespace
    return text.strip()
def get_answer(query: str, relevance_threshold: float = 1.2) -> str:
    try:
        vectordb = load_vector_store()

        if query.strip().lower() == "clear db":
            clear_vector_store("vectorstore")  # Call your clearing function here
            logging.info("Vector DB fresh start")
            return "🗑️ Vector store has been cleared."

        vectordb = load_vector_store()


        logging.info((vectordb.index.metric_type))
        docs_and_scores = vectordb.similarity_search_with_score(query, k=3)

        # Log all returned docs and scores
        for i, (doc, score) in enumerate(docs_and_scores):
            logging.info(f"[SimilaritySearch] Result {i+1}: Score={score:.4f}, Content Preview={doc.page_content[:200]}")

        relevant_docs = [
            doc for doc, score in docs_and_scores
            if score <= relevance_threshold and len(doc.page_content.strip().split()) > 20
        ]

        if relevant_docs:
            logging.info("[CHAIN] Using pdf_chain with pdf_prompt")
            context = "\n".join([doc.page_content for doc in relevant_docs]).strip()
            result = pdf_chain.invoke({"context": context, "question": query})
            return result["text"].strip()
        else:
            logging.info("[INFO] No relevant documents met the threshold (%.2f)", relevance_threshold)
        if is_greeting_query(query):
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            logging.info("[CHAIN] Using greeting_chain with greeting_prompt")
            result = greeting_chain.invoke({"query": query, "time": current_time})
            return result["text"].strip()
        logging.info(relevance_threshold)
        fallback_chain = choose_fallback_prompt(query)
        prompt_name = "explain_prompt" if fallback_chain == explain_chain else "fact_prompt"
        logging.info(f"[CHAIN] Using fallback_chain with {prompt_name}")
        result = fallback_chain.invoke(query)
        return result["text"].strip()

    except Exception as e:
        logging.exception("Final fallback also failed.")
        return "Sorry, something went wrong on the server."


def process_pdf(path: str):
    create_vector_store(path)
