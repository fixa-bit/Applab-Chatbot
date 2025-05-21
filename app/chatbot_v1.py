# # import os
# # from langchain_huggingface import HuggingFacePipeline
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# # from langchain.chains import LLMChain

# # from langchain.prompts import PromptTemplate
# # from dotenv import load_dotenv
# # # Set huggingface cache directory
# # load_dotenv()
# # #os.environ["TRANSFORMERS_CACHE"] = "d/cache"

# # def load_model():
# #     model_id = "google/flan-t5-large"
# #     tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="D:/cache")
# #     model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir="D:/cache")

# #     pipe = pipeline(
# #         "text2text-generation",
# #         model=model,
# #         tokenizer=tokenizer,
# #         max_new_tokens=200
# #     )

# #     return HuggingFacePipeline(pipeline=pipe)

# # llm = load_model()

# # # Create a general-purpose prompt
# # template = """Answer the following question as clearly and helpfully as possible:
# # Question: {question}
# # Answer:"""

# # prompt = PromptTemplate(template=template, input_variables=["question"])
# # chain = LLMChain(llm=llm, prompt=prompt)

# # def get_answer(query: str) -> str:
# #     return chain.run(query)
# import os
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from langchain.chains import RetrievalQA, LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders import TextLoader

# load_dotenv()

# # Load HuggingFace model
# def load_model():
#     model_id = "google/flan-t5-large"
#     tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="D:/cache")
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir="D:/cache")
#     pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
#     return HuggingFacePipeline(pipeline=pipe)

# llm = load_model()

# # Load documents, split, embed, and create FAISS index
# def create_vector_store(pdf_path: str):
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()

#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     chunks = splitter.split_documents(docs)

#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectordb = FAISS.from_documents(chunks, embedding=embedder)

#     vectordb.save_local("vectorstore")
#     return vectordb

# # Load existing vector store
# def load_vector_store():
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.load_local("vectorstore", embedder, allow_dangerous_deserialization=True)

# # RetrievalQA chain
# def get_rag_chain():
#     vectordb = load_vector_store()
#     return RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

# # Regular QA for general questions
# template = """Answer the following question as clearly and helpfully as possible:
# Question: {question}
# Answer:"""
# prompt = PromptTemplate(template=template, input_variables=["question"])
# general_chain = LLMChain(llm=llm, prompt=prompt)

# # Entry point
# def get_answer(query: str) -> str:
#     try:
#         rag_chain = get_rag_chain()
#         return rag_chain.run(query)
#     except:
#         return general_chain.run(query)

# def process_pdf(path: str):
#     create_vector_store(path)


import datetime
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader

load_dotenv()

# Load model
def load_model():
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="D:/cache")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir="D:/cache")
    #pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    # pipe = pipeline(
    #     "text2text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=500,      # Increased max tokens from 200 to 500
    #     do_sample=True,          # Optional: enables sampling for diversity
    #     temperature=0.7 ,         # Optional: controls randomness (0.7 is moderate)
    #     min_length=10, 
    # )
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
        min_length=50  # Ensure minimum response length
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_model()
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
def create_vector_store(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, embedding=embedder)
    vectordb.save_local("vectorstore")

# Load vector store
def load_vector_store():
    return FAISS.load_local("vectorstore", embedder, allow_dangerous_deserialization=True)


# Enhanced PDF prompt with human-like elements
pdf_prompt_template = """Respond naturally like a human expert while using the provided context to answer the question.

Context:
{context}

Question: {question}

Respond conversationally but informatively:"""
pdf_prompt = PromptTemplate(template=pdf_prompt_template, input_variables=["context", "question"])
pdf_chain = LLMChain(llm=llm, prompt=pdf_prompt)
# Prompt 2: General factual
# template_facts = """Answer this factual question clearly. Please provide a detailed and comprehensive answer.:
# Question: {question}
# Answer:"""
template_facts = """You are a knowledgeable assistant. Provide a comprehensive answer to the question including:
1. Key facts and definitions
2. Relevant context or background
3. Examples or applications (if applicable)
4. Current status or modern relevance

Question: {question}

Structured Answer:"""
fact_prompt = PromptTemplate(template=template_facts, input_variables=["question"])
fact_chain = LLMChain(llm=llm, prompt=fact_prompt)

# Prompt 3: General explanatory
template_explain = """Explain the following in simple, friendly terms as if talking to a curious friend:
1. Start with "Let me explain..."
2. Break it down clearly
3. Use relatable examples
4. Check for understanding

Example format:
"Let me break this down for you! [Simple definition]. 
For example, [real-world example]. 
Does this make sense so far?"

Topic: {question}

Friendly explanation:"""

explain_prompt = PromptTemplate(template=template_explain, input_variables=["question"])
explain_chain = LLMChain(llm=llm, prompt=explain_prompt)


greeting_prompt_template = '''You are a virtual assistant. Respond to greetings or casual inputs in a professional  way.

Guidelines:
- Keep it short and to the point.
- Include only one relevant emoji (if appropriate).
- Do NOT include any personal information
- Be context-aware (consider time of day).
- Answer the question asked (if any question)

Input: {query}
Current time: {time}
Response:'''
greeting_prompt = PromptTemplate(
    template=greeting_prompt_template,
    input_variables=["query", "time"]
)
greeting_chain = LLMChain(llm=llm, prompt=greeting_prompt)

# Enhanced general factual prompt
template_facts = """You are a knowledgeable assistant who speaks naturally. Structure your response:
1. Start with a friendly acknowledgment
2. Provide core facts conversationally
3. Add interesting details
4. End with an inviting question

Example format:
"That's a great question about [topic]! [Main answer]. 
Did you know [interesting fact]? 
Would you like more details about any aspect?"

Question: {question}

Natural but informative answer:"""
fact_prompt = PromptTemplate(template=template_facts, input_variables=["question"])

# Logic to select fallback prompt
def choose_fallback_prompt(query: str):
    if any(word in query.lower() for word in ["how", "why", "explain", "difference", "what is"]):
        return explain_chain
    return fact_chain

# Main logic
def get_answer2(query: str) -> str:
    try:
        vectordb = load_vector_store()
        retriever = vectordb.as_retriever()
        docs = retriever.get_relevant_documents(query)

        if docs:
            context = "\n".join([doc.page_content for doc in docs[:3]])
            return pdf_chain.run({"context": context, "question": query})
        else:
            selected_chain = choose_fallback_prompt(query)
            return selected_chain.run(query)

    except Exception:
        # If vector DB fails
        fallback_chain = choose_fallback_prompt(query)
        return fallback_chain.run(query)
# def get_answer(query: str) -> str:
#     try:
#         vectordb = load_vector_store()
#         retriever = vectordb.as_retriever()
#         docs = retriever.get_relevant_documents(query)

#         # Use PDF context if available
#         if docs:
#             context = "\n".join([doc.page_content for doc in docs[:3]]).strip()
#             if context:
#                 return pdf_chain.run({"context": context, "question": query})

#         # Fallback to explanatory prompt only
#         return explain_chain.run(query)

#     except Exception:
#         # If anything fails (e.g., vectorstore not found), fallback to explanatory prompt
#         return explain_chain.run(query)




# def get_answer(query: str, relevance_threshold: float = 0.7) -> str:
#     try:
#         vectordb = load_vector_store()
#         retriever = vectordb.as_retriever(search_type="similarity_score")  # enable scoring if supported

#         # Get docs with similarity scores (list of (doc, score))
#         docs_and_scores = vectordb.similarity_search_with_score(query, k=3)
        

#         # Filter docs above threshold
#         relevant_docs = [doc for doc, score in docs_and_scores if score >= relevance_threshold]

#         if relevant_docs:
#             context = "\n".join([doc.page_content for doc in relevant_docs]).strip()
#             if context:
#                 return pdf_chain.run({"context": context, "question": query})

#         # Fallback to explanatory prompt if no relevant doc found
#         return explain_chain.run(query)

#     except Exception:
#         return explain_chain.run(query)

from datetime import datetime
def get_answer(query: str, relevance_threshold: float = 0.7) -> str:
    # First check if it's a general conversation/greeting
    conversational_phrases = ["hi", "hello", "hey", "how are you", "what's up", 
                            "thanks", "thank you", "good morning", "good afternoon", 
                            "good evening"]
    if any(phrase in query.lower() for phrase in conversational_phrases):
        current_time = datetime.now().strftime("%H:%M")
        return greeting_chain.run({"query": query, "time": current_time})
    
    try:
        # Check for relevant documents first
        vectordb = load_vector_store()
        docs_and_scores = vectordb.similarity_search_with_score(query, k=3)
        
        # Filter relevant docs
        relevant_docs = [doc for doc, score in docs_and_scores if score >= relevance_threshold]
        
        if relevant_docs:
            context = "\n".join([doc.page_content for doc in relevant_docs]).strip()
            if context:
                return pdf_chain.run({"context": context, "question": query})
        
        # If no relevant docs, check if it's an explanatory question
        if any(word in query.lower() for word in ["how", "why", "explain", "what is", "describe"]):
            return explain_chain.run(query)
        
        # Default to factual chain for other queries
        return fact_chain.run(query)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Fallback to factual chain on error
        return fact_chain.run(query)



# PDF processor
def process_pdf(path: str):
    create_vector_store(path)
