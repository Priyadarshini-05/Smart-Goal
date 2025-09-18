import streamlit as st
import fitz  # PyMuPDF
import re
import os
import tempfile
import numpy as np
from typing import List, Tuple
import faiss
import requests
from io import BytesIO
import json

# Set page configuration
st.set_page_config(
    page_title="StudyMate - AI Academic Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        max-height: 60vh;
        overflow-y: auto;
    }
    .user-message {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .assistant-message {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
    }
    .page-reference {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "default"
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "page_mapping" not in st.session_state:
    st.session_state.page_mapping = []

# Simple embedding function (replace with proper embeddings if needed)
def simple_embedding(text_chunks):
    # This is a simple placeholder for actual embeddings
    # In a real implementation, you'd use a proper embedding model
    embeddings = []
    for chunk in text_chunks:
        # Simple heuristic: count character frequencies as a basic embedding
        embedding = np.zeros(256)  # Using 256 dimensions for simplicity
        for char in chunk:
            if ord(char) < 256:
                embedding[ord(char)] += 1
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    return np.array(embeddings)

# Process PDF documents
def process_pdfs(uploaded_files):
    text_chunks = []
    page_mapping = []
    
    for file_idx, uploaded_file in enumerate(uploaded_files):
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        doc = None
        try:
            # Open the PDF
            doc = fitz.open(tmp_path)
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Split text into chunks (by sentences for better context)
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|?)\s', text)
                chunk = ""
                for sentence in sentences:
                    if len(chunk) + len(sentence) < 500:  # Rough chunk size limit
                        chunk += sentence + " "
                    else:
                        if chunk.strip():
                            text_chunks.append(chunk.strip())
                            page_mapping.append((uploaded_file.name, page_num + 1))
                        chunk = sentence + " "
                
                # Add the last chunk for the page
                if chunk.strip():
                    text_chunks.append(chunk.strip())
                    page_mapping.append((uploaded_file.name, page_num + 1))
        
        finally:
            if doc is not None:
                doc.close()
            
            try:
                os.unlink(tmp_path)
            except PermissionError:
                # If we can't delete immediately, try again after a short delay
                import time
                time.sleep(0.1)
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    # If still can't delete, log the issue but continue
                    st.warning(f"Could not delete temporary file {tmp_path}. This may cause disk space issues.")
    
    return text_chunks, page_mapping

# Create FAISS index
def create_faiss_index(text_chunks):
    # Generate simple embeddings
    embeddings = simple_embedding(text_chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    return index, embeddings

# Search for relevant chunks
def search_relevant_chunks(query, index, chunks, page_mapping, k=3):
    # Generate query embedding using the same simple method
    query_embedding = simple_embedding([query])
    distances, indices = index.search(query_embedding, k)
    
    relevant_chunks = []
    relevant_pages = []
    
    for idx in indices[0]:
        if idx < len(chunks):  # Ensure index is within bounds
            relevant_chunks.append(chunks[idx])
            relevant_pages.append(page_mapping[idx])
    
    return relevant_chunks, relevant_pages

# Generate answer using the model
def generate_answer(query, context_chunks):
    """
    Simple answer generation without external models.
    In a production environment, you could replace this with API calls to OpenAI, Anthropic, etc.
    """
    # Combine context chunks
    context = " ".join(context_chunks[:3])
    
    # Simple keyword-based response generation
    query_lower = query.lower()
    
    # Extract key sentences from context that might be relevant
    sentences = re.split(r'[.!?]+', context)
    relevant_sentences = []
    
    # Look for sentences containing query keywords
    query_words = set(query_lower.split())
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        if query_words.intersection(sentence_words):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        # Return the most relevant sentences
        answer = ". ".join(relevant_sentences[:2])
        if not answer.endswith('.'):
            answer += "."
        return answer
    else:
        # Fallback response
        return f"Based on the uploaded documents, I found relevant information but cannot provide a specific answer to '{query}'. Please try rephrasing your question or check if the information is available in your documents."

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">StudyMate</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">AI-Powered Academic Assistant</h2>', unsafe_allow_html=True)
    
    # Sidebar for chat management and document upload
    with st.sidebar:
        st.header("Chat Sessions")
        
        # Create new chat button
        if st.button("+ New Chat"):
            chat_name = f"Chat_{len(st.session_state.chat_history) + 1}"
            st.session_state.chat_history[chat_name] = []
            st.session_state.current_chat = chat_name
        
        # Display existing chats
        chat_list = list(st.session_state.chat_history.keys())
        if chat_list:
            selected_chat = st.selectbox("Select Chat", chat_list, 
                                        index=chat_list.index(st.session_state.current_chat) if st.session_state.current_chat in chat_list else 0)
            st.session_state.current_chat = selected_chat
        
        st.divider()
        
        # Document upload section
        st.header("Upload Study Materials")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files and not st.session_state.documents_processed:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    # Process PDFs
                    chunks, page_mapping = process_pdfs(uploaded_files)
                    
                    # Create FAISS index
                    index, embeddings = create_faiss_index(chunks)
                    
                    # Store in session state
                    st.session_state.faiss_index = index
                    st.session_state.chunks = chunks
                    st.session_state.page_mapping = page_mapping
                    st.session_state.embeddings = embeddings
                    st.session_state.documents_processed = True
                    
                st.success(f"Documents processed successfully! Created {len(chunks)} text chunks.")
    
    # Main chat area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history for current chat
    if st.session_state.current_chat in st.session_state.chat_history:
        for message in st.session_state.chat_history[st.session_state.current_chat]:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                # Display answer with page references if available
                content = message["content"]
                if "pages" in message and message["pages"]:
                    page_refs = ", ".join([f"{p[0]} (Page {p[1]})" for p in message["pages"]])
                    content += f'<div class="page-reference">References: {page_refs}</div>'
                st.markdown(f'<div class="assistant-message">{content}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input at bottom
    query = st.chat_input("Ask a question about your documents...")
    
    if query:
        # Add user message to chat history
        if st.session_state.current_chat not in st.session_state.chat_history:
            st.session_state.chat_history[st.session_state.current_chat] = []
        
        st.session_state.chat_history[st.session_state.current_chat].append({
            "role": "user",
            "content": query
        })
        
        # Display user message immediately
        st.markdown(f'<div class="user-message">{query}</div>', unsafe_allow_html=True)
        
        # Generate response
        if st.session_state.documents_processed:
            with st.spinner("Thinking..."):
                # Search for relevant chunks
                relevant_chunks, relevant_pages = search_relevant_chunks(
                    query, 
                    st.session_state.faiss_index, 
                    st.session_state.chunks, 
                    st.session_state.page_mapping
                )
                
                # Generate answer
                answer = generate_answer(query, relevant_chunks)
                
                # Add assistant message to chat history
                st.session_state.chat_history[st.session_state.current_chat].append({
                    "role": "assistant",
                    "content": answer,
                    "pages": relevant_pages
                })
                
                # Display assistant message with page references
                page_refs = ", ".join([f"{p[0]} (Page {p[1]})" for p in relevant_pages])
                answer_html = f"{answer}<div class='page-reference'>References: {page_refs}</div>"
                st.markdown(f'<div class="assistant-message">{answer_html}</div>', unsafe_allow_html=True)
        else:
            # No documents processed yet
            warning_msg = "Please upload and process documents first to get answers based on your materials."
            st.session_state.chat_history[st.session_state.current_chat].append({
                "role": "assistant",
                "content": warning_msg
            })
            st.markdown(f'<div class="assistant-message">{warning_msg}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
