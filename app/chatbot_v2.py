# import datetime
# import os
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.document_loaders import PyPDFLoader

# load_dotenv()

# # Load model
# def get_pipeline(mode: str, model, tokenizer):
#     # model_id = "google/flan-t5-large"
#     # tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="D:/cache")
#     # model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir="D:/cache")
#     #pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
#     # pipe = pipeline(
#     #     "text2text-generation",
#     #     model=model,
#     #     tokenizer=tokenizer,
#     #     max_new_tokens=500,      # Increased max tokens from 200 to 500
#     #     do_sample=True,          # Optional: enables sampling for diversity
#     #     temperature=0.7 ,         # Optional: controls randomness (0.7 is moderate)
#     #     min_length=10, 
#     # )
#     # pipe = pipeline(
#     #     "text2text-generation",
#     #     model=model,
#     #     tokenizer=tokenizer,
#     #     max_new_tokens=500,
#     #     do_sample=True,
#     #     temperature=0.8,
#     #     top_k=50,
#     #     top_p=0.95,
#     #     repetition_penalty=1.2,
#     #     no_repeat_ngram_size=3,
#     #     early_stopping=True,
#     #     min_length=50  # Ensure minimum response length
#     # )
#     # return HuggingFacePipeline(pipeline=pipe)
#     if mode=="precise":
#         pipe = pipeline(
#             "text2text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=300,            # Slightly lower = tighter, focused response
#             do_sample=False,               # Disable randomness = more deterministic
#             temperature=0.0,               # No randomness = avoid hallucination
#             repetition_penalty=1.1,        # Penalize repetition
#             no_repeat_ngram_size=3,        # Avoid phrase repetition
#             early_stopping=True
#         )


#     elif mode == "creative":
#         pipe = pipeline(
#             "text2text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=500,
#             do_sample=True,
#             temperature=0.8,
#             top_k=50,
#             top_p=0.95,
#             repetition_penalty=1.2,
#             no_repeat_ngram_size=3,
#             early_stopping=True,
#             min_length=50
#         )
#     elif mode == "safe":
#         pipe = pipeline(
#             "text2text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=30,
#             do_sample=False,
#             temperature=0.3,
#             repetition_penalty=1.1,
#             no_repeat_ngram_size=2,
#             early_stopping=True
#         )
#     else:
#         raise ValueError("Unsupported mode")
    
#     return HuggingFacePipeline(pipeline=pipe)

# def load_model_and_tokenizer():
#     model_id = "google/flan-t5-large"
#     tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="D:/cache")
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir="D:/cache")
#     return model, tokenizer


# # Enhanced PDF prompt with human-like elements
# pdf_prompt_template = """You are a helpful assistant. Use the context to answer the question.
#         If the answer isn't in the context, reply accordingly.
# Context:
# {context}

# Question: {question}

# Respond conversationally but informatively:"""
# pdf_prompt = PromptTemplate(template=pdf_prompt_template, input_variables=["context", "question"])
# #pdf_chain = LLMChain(llm=llm, prompt=pdf_prompt)
# # Prompt 2: General factual
# # template_facts = """Answer this factual question clearly. Please provide a detailed and comprehensive answer.:
# # Question: {question}
# # Answer:"""
# template_facts = """You are a knowledgeable assistant. Provide a comprehensive answer to the question including:
# 1. Key facts and definitions
# 2. Relevant context or background
# 3. Examples or applications (if applicable)
# 4. Current status or modern relevance

# Question: {question}

# Structured Answer:"""
# fact_prompt = PromptTemplate(template=template_facts, input_variables=["question"])
# #fact_chain = LLMChain(llm=llm, prompt=fact_prompt)

# # Prompt 3: General explanatory
# template_explain = """Explain the following in simple, friendly terms give to the point accurate information:
# 1. Start with "Let me explain..."
# 2. Break it down clearly
# 3. Use relatable examples
# 4. Check for understanding

# Example format:
# "Let me break this down for you! [Simple definition]. 
# For example, [real-world example]. 
# Does this make sense so far?"

# Topic: {question}

# Friendly explanation:"""

# explain_prompt = PromptTemplate(template=template_explain, input_variables=["question"])
# #explain_chain = LLMChain(llm=llm, prompt=explain_prompt)


# greeting_prompt_template = """You are a friendly assistant. Respond to greetings or casual inputs in a warm, natural way.

# Guidelines:
# - Keep it short and polite.
# - Include only one relevant emoji (if appropriate).
# - Do NOT include usernames, emails, social media handles, or sign-offs like "xoxo".
# - Be context-aware (consider time of day).

# Input: {query}
# Current time: {time}

# Polite and simple response:"""
# greeting_prompt = PromptTemplate(
#     template=greeting_prompt_template,
#     input_variables=["query", "time"]
# )
# #greeting_chain = LLMChain(llm=llm, prompt=greeting_prompt)

# # Enhanced general factual prompt
# template_facts = """You are a knowledgeable assistant who speaks naturally. Structure your response:
# 1. Start with a friendly acknowledgment
# 2. Provide core facts conversationally
# 3. Add interesting details
# 4. End with an inviting question

# Example format:
# "That's a great question about [topic]! [Main answer]. 
# Did you know [interesting fact]? 
# Would you like more details about any aspect?"

# Question: {question}

# Natural but informative answer:"""
# fact_prompt = PromptTemplate(template=template_facts, input_variables=["question"])





# model, tokenizer = load_model_and_tokenizer()
# llm_creative = get_pipeline("creative", model, tokenizer)
# llm_safe = get_pipeline("safe", model, tokenizer)
# llm_pdf = get_pipeline("precise", model, tokenizer)

# pdf_chain = LLMChain(llm=llm_pdf, prompt=pdf_prompt)
# fact_chain = LLMChain(llm=llm_creative, prompt=fact_prompt)
# explain_chain = LLMChain(llm=llm_creative, prompt=explain_prompt)
# greeting_chain = LLMChain(llm=llm_safe, prompt=greeting_prompt)
# #llm = load_model()
# embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Create vector store
# def create_vector_store(pdf_path: str):
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     chunks = splitter.split_documents(docs)
#     vectordb = FAISS.from_documents(chunks, embedding=embedder)
#     vectordb.save_local("vectorstore")

# # Load vector store
# def load_vector_store():
#     return FAISS.load_local("vectorstore", embedder, allow_dangerous_deserialization=True)



# # Logic to select fallback prompt
# def choose_fallback_prompt(query: str):
#     if any(word in query.lower() for word in ["how", "why", "explain", "difference", "what is"]):
#         return explain_chain
#     return fact_chain

# # Main logic
# def get_answer2(query: str) -> str:
#     try:
#         vectordb = load_vector_store()
#         retriever = vectordb.as_retriever()
#         docs = retriever.get_relevant_documents(query)

#         if docs:
#             context = "\n".join([doc.page_content for doc in docs[:3]])
#             return pdf_chain.run({"context": context, "question": query})
#         else:
#             selected_chain = choose_fallback_prompt(query)
#             return selected_chain.run(query)

#     except Exception:
#         # If vector DB fails
#         fallback_chain = choose_fallback_prompt(query)
#         return fallback_chain.run(query)
# # def get_answer(query: str) -> str:
# #     try:
# #         vectordb = load_vector_store()
# #         retriever = vectordb.as_retriever()
# #         docs = retriever.get_relevant_documents(query)

# #         # Use PDF context if available
# #         if docs:
# #             context = "\n".join([doc.page_content for doc in docs[:3]]).strip()
# #             if context:
# #                 return pdf_chain.run({"context": context, "question": query})

# #         # Fallback to explanatory prompt only
# #         return explain_chain.run(query)

# #     except Exception:
# #         # If anything fails (e.g., vectorstore not found), fallback to explanatory prompt
# #         return explain_chain.run(query)




# # def get_answer(query: str, relevance_threshold: float = 0.7) -> str:
# #     try:
# #         vectordb = load_vector_store()
# #         retriever = vectordb.as_retriever(search_type="similarity_score")  # enable scoring if supported

# #         # Get docs with similarity scores (list of (doc, score))
# #         docs_and_scores = vectordb.similarity_search_with_score(query, k=3)
        

# #         # Filter docs above threshold
# #         relevant_docs = [doc for doc, score in docs_and_scores if score >= relevance_threshold]

# #         if relevant_docs:
# #             context = "\n".join([doc.page_content for doc in relevant_docs]).strip()
# #             if context:
# #                 return pdf_chain.run({"context": context, "question": query})

# #         # Fallback to explanatory prompt if no relevant doc found
# #         return explain_chain.run(query)

# #     except Exception:
# #         return explain_chain.run(query)
# def is_greeting_query(query: str) -> bool:
#     greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
#     return any(greet in query.lower() for greet in greetings)

# def get_answer(query: str, relevance_threshold: float = 0.9) -> str:
#     try:
#         # 1. Greeting Check
#         if is_greeting_query(query):
#             current_time = datetime.datetime.now().strftime("%I:%M %p")
#             return greeting_chain.run({"query": query, "time": current_time})

#         # 2. PDF Vector Retrieval
#         vectordb = load_vector_store()
#         docs_and_scores = vectordb.similarity_search_with_score(query, k=3)

#         # 3. Filter relevant docs
#         # relevant_docs = [doc for doc, score in docs_and_scores if score >= relevance_threshold]
#         relevant_docs = [
#             doc for doc, score in docs_and_scores
#             if score >= relevance_threshold and len(doc.page_content.strip().split()) > 20
#         ]

#         # 4. Use PDF context if available
#         if relevant_docs:
#             context = "\n".join([doc.page_content for doc in relevant_docs]).strip()
#             # if "@" in context or "http" in context or len(context) < 30:
#             #     return choose_fallback_prompt(query).run(query)
#             return pdf_chain.run({"context": context, "question": query})

#         # 5. Fallback: use factual or explanatory depending on question
#         fallback_chain = choose_fallback_prompt(query)
#         return fallback_chain.run(query)

#     except Exception:
#         # 6. Error fallback
#         fallback_chain = choose_fallback_prompt(query)
#         return fallback_chain.run(query)


# # PDF processor
# def process_pdf(path: str):
#     create_vector_store(path)



import datetime
import os
import logging
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFacePipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv()

def get_pipeline(mode: str, model, tokenizer):
    logging.info(f"Initializing LLM pipeline with mode: {mode}")
    if mode == "precise":
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    elif mode == "creative":
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            #num_beams=4, 
            early_stopping=True,
            min_length=50
        )
    elif mode == "safe":
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.4,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            #num_beams=4, 
            early_stopping=True
        )
    else:
        raise ValueError("Unsupported mode")

    return HuggingFacePipeline(pipeline=pipe)

def load_model_and_tokenizer():
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="D:/cache")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir="D:/cache")
    return model, tokenizer

# Prompt templates
pdf_prompt_template = """You are a helpful assistant. Use the context to answer the question.
        If the answer isn't in the context, reply accordingly.
Context:
{context}

Question: {question}

Respond conversationally but informatively:"""
pdf_prompt = PromptTemplate(template=pdf_prompt_template, input_variables=["context", "question"])

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

template_explain = """Explain the following in simple, friendly terms give to the point accurate information:
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

greeting_prompt_template = """You are a friendly assistant. Respond to greetings or casual inputs in a warm, natural way.

Guidelines:
- Keep it short and polite.
- Include only one relevant emoji (if appropriate).
- Do NOT include usernames, emails, social media handles, or sign-offs like "xoxo".
- Be context-aware (consider time of day).

Input: {query}
Current time: {time}

Response:"""
greeting_prompt = PromptTemplate(template=greeting_prompt_template, input_variables=["query", "time"])

# Load model and create LLM chains
model, tokenizer = load_model_and_tokenizer()
llm_creative = get_pipeline("creative", model, tokenizer)
llm_safe = get_pipeline("safe", model, tokenizer)
llm_pdf = get_pipeline("precise", model, tokenizer)

pdf_chain = LLMChain(llm=llm_pdf, prompt=pdf_prompt)
fact_chain = LLMChain(llm=llm_creative, prompt=fact_prompt)
explain_chain = LLMChain(llm=llm_creative, prompt=explain_prompt)
greeting_chain = LLMChain(llm=llm_safe, prompt=greeting_prompt)

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector Store
def create_vector_store(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, embedding=embedder)
    vectordb.save_local("vectorstore")

def load_vector_store():
    return FAISS.load_local("vectorstore", embedder, allow_dangerous_deserialization=True)

def choose_fallback_prompt(query: str):
    if any(word in query.lower() for word in ["how", "why", "explain", "difference", "what is"]):
        logging.info("Selected prompt: explain_chain")
        return explain_chain
    logging.info("Selected prompt: fact_chain")
    return fact_chain

def is_greeting_query(query: str) -> bool:
    query = query.lower().strip().split()
    greetings = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"}
    #greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    return any(greet in query.lower() for greet in greetings)

def get_answer(query: str, relevance_threshold: float = 0.9) -> str:
    try:
        if is_greeting_query(query):
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            logging.info("Running greeting_chain with greeting_prompt")
            return greeting_chain.invoke({"query": query, "time": current_time})

        vectordb = load_vector_store()
        docs_and_scores = vectordb.similarity_search_with_score(query, k=3)

        relevant_docs = [
            doc for doc, score in docs_and_scores
            if score >= relevance_threshold and len(doc.page_content.strip().split()) > 20
        ]

        if relevant_docs:
            context = "\n".join([doc.page_content for doc in relevant_docs]).strip()
            logging.info("Running pdf_chain with pdf_prompt")
            return pdf_chain.invoke({"context": context, "question": query})

        fallback_chain = choose_fallback_prompt(query)
        logging.info(f"Running fallback_chain with prompt: {fallback_chain.prompt.template[:60]}...")
        return fallback_chain.invoke(query)

    except Exception as e:
        logging.warning(f"Exception occurred: {e}. Falling back.")
        fallback_chain = choose_fallback_prompt(query)
        return fallback_chain.invoke(query)

def process_pdf(path: str):
    create_vector_store(path)
