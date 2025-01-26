# Importing Libraries
import faiss
import PyPDF2
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings("ignore")


# Configure the Embedding Model and FAISS for Vector-based Retrieval
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)
metadata = {}


# Streamlit UI and API Key Input
st.title("VidyaVistaar: The MultiLingual Educational Resource Generator App")


# Step 1: API Key Input
st.header("Step 1: Enter Your Google Gemini API Key")
api_key = st.text_input("Enter your API Key:", type="password")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    st.success("API Key authenticated successfully!")
else:
    st.warning("Please enter a valid API Key to continue.")


# Extract texts from the pdf
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Splits text into sentence-aware chunks with overlap.
def split_text_into_chunks(text, max_chunk_size=2000, overlap=50):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = current_chunk[-overlap:] + sentence
        else:
            current_chunk += ". " + sentence if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# Generates embeddings for text chunks and stores them in FAISS.
def generate_and_store_embeddings(text, doc_id="doc1"):
    chunks = split_text_into_chunks(text)
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype(np.float32)
    index.add(embeddings) # Add embeddings to FAISS index

     # Store metadata for each embedding
    for i, chunk in enumerate(chunks):
        metadata[len(metadata)] = {"chunk": chunk, "doc_id": doc_id}


# Retrieves the top-k most relevant chunks for a query.
def retrieve_chunks(query, top_k=5):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype(np.float32)
    distances, indices = index.search(query_embedding, top_k)
    results = [{"chunk": metadata[idx]["chunk"], "doc_id": metadata[idx]["doc_id"]} for idx in indices[0] if idx != -1]
    return results


# Generate Course with RAG
def generate_rescontent(query, retrieved_chunks, prompt):
    if not retrieved_chunks:
        return "Sorry, I couldn't find any relevant information for your request."
 
    retrieved_context = " ".join([chunk["chunk"] for chunk in retrieved_chunks])
    full_prompt = f"{prompt}\n\n{retrieved_context}\n\nGenerated Content:"
    response = model.generate_content(full_prompt)
    return response.text.strip()


# Generates a response For Chatbot using Google Gemini.
def generate_response(query, retrieved_chunks):
    if not retrieved_chunks:
      return "Sorry, I couldn't find any relevant information for your question."

    retrieved_context = " ".join([chunk["chunk"] for chunk in retrieved_chunks])
    prompt = f"Context: {retrieved_context}\n\nQuestion: {query}\n\nAnswer:"
    response = model.generate_content(prompt)
    return response.text.strip()


# Translate the response before giving the Output.
def translate_text(text, language_name):
    prompt = f"Translate the following text to {language_name}: {text}"
    response = model.generate_content(prompt)
    return response.text.strip()


# Various Prompts for Various Tasks.
def create_prompts(text, task_type):
    prompts = {
        "mcq": f"Read the following text carefully and generate multiple-choice questions. Each question should include:\n"
               f"1. A clear and concise question based on the text.\n"
               f"2. Give a question with Four options (A, B, C, D), with one correct answer clearly indicated.\n"
               f"3. The questions should cover key concepts, definitions, critical points, and significant details discussed in the text.\n"
               f"4. Ensure the options are plausible and relevant to the content.\n\n"
               f"Text:\n{text}\n\nMCQ:",
        "fill_in_the_blank": f"Read the following text thoroughly and generate fill-in-the-blank questions. Each question should include:\n"
                            f"1. A sentence from the text with one key term or concept replaced by a blank.\n"
                            f"2. The correct term or concept that completes the sentence accurately.\n"
                            f"3. Focus on important information, such as key terms, dates, names, and concepts that are critical to understanding the text.\n\n"
                            f"Text:\n{text}\n\nFill in the blank:",
        "short_answer": f"Read the following text attentively and generate short answer questions. Each question should include:\n"
                        f"1. A clear and specific question that requires a brief response.\n"
                        f"2. The response should address key points, explanations, or definitions provided in the text.\n"
                        f"3. Ensure the questions encourage critical thinking and comprehension of the material, focusing on important details and concepts.\n\n"
                        f"Text:\n{text}\n\nShort answer question:",
        "course": f"Read the following text and generate a comprehensive, structured curriculum content. The content should include:\n"
                  f"1. Learning objectives and outcomes.\n"
                  f"2. Topic-wise breakdown with detailed descriptions.\n"
                  f"3. Key concepts, definitions, and explanations.\n"
                  f"4. Examples, illustrations, and case studies.\n"
                  f"5. Assessment and evaluation criteria.\n\n"
                  f"Text:\n{text}\n\nCurriculum Content:",
    }
    return prompts.get(task_type, "")


# Step 2: Upload PDF File
st.header("Step 2: Upload Your PDF File")
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if pdf_file:
    pdf_text = extract_text_from_pdf(pdf_file)
    generate_and_store_embeddings(pdf_text, doc_id="uploaded_pdf")
    st.success("PDF processed successfully! You can now ask questions or generate content.")

    # Step 3: Select Language
    st.header("Step 3: Select Output Language")
    languages = [
        "Arabic", "Czech", "German", "English", "Spanish", "Estonian", "Finnish", "French", "Gujarati",
        "Hindi", "Italian", "Japanese", "Kazakh", "Korean", "Lithuanian", "Latvian", "Burmese", "Nepali",
        "Dutch", "Romanian", "Russian", "Sinhala", "Turkish", "Vietnamese", "Chinese", "Afrikaans",
        "Azerbaijani", "Bengali", "Persian", "Hebrew", "Croatian", "Indonesian", "Georgian", "Khmer",
        "Macedonian", "Malayalam", "Mongolian", "Marathi", "Polish", "Pashto", "Portuguese", "Swedish",
        "Swahili", "Tamil", "Telugu", "Thai", "Tagalog", "Ukrainian", "Urdu", "Xhosa", "Galician",
        "Slovene"
    ]
    language_choice = st.selectbox("Choose a language for the output:", languages)

    # Step 4: Main Operations
    st.header("Step 4: Choose an Operation")
    options = ["Generate Course", "Generate Questions", "Chat with PDF"]
    choice = st.radio("Select an operation:", options)

    if choice == "Generate Course":
        st.subheader("A Comprehensive Course Outline")
        prompt = create_prompts(pdf_text, "course")
        retrieved_chunks = retrieve_chunks("Generate a comprehensive course outline based on the following content.", top_k=5)
        course_content = generate_rescontent("Generate a course based on the content.", retrieved_chunks, prompt)
        translated_content = translate_text(course_content, language_choice)
        st.write(translated_content)

    elif choice == "Generate Questions":
        st.subheader("Generate Questions")
        question_type = st.selectbox("Choose the type of questions:", ["MCQ", "Fill in the Blank", "Short Answer"])
        num_questions = st.slider("Number of questions to generate:", 1, 15, 5)
        retrieved_chunks = retrieve_chunks(f"Generate {question_type.lower()} questions based on the content.", top_k=5)
        prompt = create_prompts(" ".join([chunk["chunk"] for chunk in retrieved_chunks]), question_type)
        questions = []

        for _ in range(num_questions):
            question = generate_rescontent(f"Generate a {question_type.lower()} question based on the context.", retrieved_chunks, prompt)
            translated_question = translate_text(question, language_choice)
            questions.append(translated_question)

        for idx, question in enumerate(questions, 1):
            st.write(f"{idx}. {question}")

    elif choice == "Chat with PDF":
        st.subheader("Chat with the PDF")
        user_query = st.text_input("Ask a question about the content of the PDF:", key="chat_pdf_query")
        if user_query:
            retrieved_chunks = retrieve_chunks(user_query, top_k=5)
            response = generate_response(user_query, retrieved_chunks)
            translated_answer = translate_text(response, language_choice)
            st.write(f"Answer: \n{translated_answer}")
