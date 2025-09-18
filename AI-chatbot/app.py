import streamlit as st
import os
from typing import List, Dict
import time
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64
import requests

# Import our custom modules
from pdf_processor import PDFProcessor
from embedding_search import EmbeddingSearchSystem
from llm_integration import LLMManager
from display_sources import display_sources
from image_generator import ImageGenerator  # Added image generator import

# Page configuration
st.set_page_config(
    page_title="StudyMate - AI Academic Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #1565c0;
    }
    .assistant-message {
        background-color: #1a1a1a;
        border-left: 4px solid #4caf50;
        color: #ffffff;
    }
    .stats-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    
    if 'search_system' not in st.session_state:
        st.session_state.search_system = EmbeddingSearchSystem()
    
    if 'llm_integration' not in st.session_state:
        st.session_state.llm_integration = LLMManager()
    
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'file_mapping' not in st.session_state:
        st.session_state.file_mapping = {}
    
    if 'followup_questions' not in st.session_state:
        st.session_state.followup_questions = []
    
    if 'chat_session_id' not in st.session_state:
        st.session_state.chat_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    
    if 'image_generator' not in st.session_state:
        st.session_state.image_generator = ImageGenerator()

def display_header():
    """Display the main header"""
    st.markdown('<div class="main-header">📚 StudyMate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Academic Assistant for PDF Documents</div>', unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with file upload and settings"""
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to start asking questions"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                process_documents(uploaded_files)
        
        # Display document statistics if available
        if st.session_state.documents_processed:
            display_document_stats()
        
        # Settings section
        st.header("⚙️ Settings")
        
        # Search settings
        st.subheader("Search Settings")
        max_chunks = st.slider("Max context chunks", 1, 10, 3)
        min_similarity = st.slider("Min similarity score", 0.0, 1.0, 0.3, 0.1)
        
        # Store settings in session state
        st.session_state.max_chunks = max_chunks
        st.session_state.min_similarity = min_similarity
        
        
        
        
        # Chat statistics
        if st.session_state.chat_history:
            st.metric("Questions Asked", st.session_state.question_count)
            st.metric("Chat Messages", len(st.session_state.chat_history))
        
        if st.session_state.chat_history:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Export PDF"):
                    export_chat_as_pdf()
            with col2:
                if st.button("📥 Export JSON"):
                    export_chat_history()
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History", type="secondary"):
                clear_chat_history()

def process_documents(uploaded_files):
    """Process uploaded PDF documents"""
    with st.spinner("Processing documents..."):
        try:
            # Process PDFs
            chunks, file_mapping = st.session_state.pdf_processor.process_uploaded_files(uploaded_files)
            
            # Build search index
            st.session_state.search_system.build_index(chunks)
            
            # Update session state
            st.session_state.documents_processed = True
            st.session_state.file_mapping = file_mapping
            
            st.success(f"✅ Processed {len(uploaded_files)} documents with {len(chunks)} chunks")
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")

def display_document_stats():
    """Display statistics about processed documents"""
    if st.session_state.documents_processed:
        stats = st.session_state.search_system.get_statistics()
        
        st.markdown("### 📊 Document Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files", stats.get('total_files', 0))
            st.metric("Pages", stats.get('total_pages', 0))
        
        with col2:
            st.metric("Chunks", stats.get('total_chunks', 0))
            st.metric("Avg Length", f"{stats.get('average_chunk_length', 0):.0f}")

def display_chat_interface():
    """Display the main chat interface"""
    st.header("💬 Ask Questions About Your Documents")
    
    if not st.session_state.documents_processed:
        st.info("👆 Please upload and process PDF documents first to start asking questions.")
        return
    
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>StudyMate:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
                
                # Display sources if available
                if 'sources' in message and message['sources']:
                    display_sources(message['sources'])
                
                # Display image if available
                if 'image' in message and message['image']:
                    st.image(f"data:image/png;base64,{message['image']}", use_column_width=True)
    
    st.markdown("---")
    
    # Question input
    question = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What are the main concepts discussed in chapter 3?",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        ask_button = st.button("Ask Question", type="primary")
    with col2:
        if st.session_state.chat_history:
            if st.button("🔄 Regenerate Last"):
                regenerate_last_answer()
    with col3:
        generate_image_button = st.button("🎨 Generate Image")
    
    if ask_button and question:
        handle_question(question)
    
    if generate_image_button and question:
        handle_image_generation(question)

def handle_question(question: str):
    """Handle user question and generate response"""
    with st.spinner("Searching documents and generating answer..."):
        try:
            print(f"[v0] Processing question: {question}")
            
            # Get relevant context
            context, source_refs = st.session_state.search_system.get_context_for_query(
                question,
                max_chunks=st.session_state.get('max_chunks', 3),
                min_score=st.session_state.get('min_similarity', 0.3)
            )
            
            print(f"[v0] Found context length: {len(context)}, sources: {len(source_refs)}")
            
            # Generate answer
            result = st.session_state.llm_integration.generate_answer(
                question, context, source_refs
            )
            
            print(f"[v0] LLM result: {result.get('success', False)}, answer length: {len(result.get('answer', ''))}")
            
            if not result.get('success', True):
                error_message = result.get('error', 'Unknown error occurred')
                answer = f"I encountered an issue generating an answer: {error_message}\n\nHowever, I found some relevant content from your documents:\n\n{context[:500]}..."
                result['answer'] = answer
                result['success'] = True  # Mark as success so it displays
            
            if not result.get('answer', '').strip():
                if context.strip():
                    result['answer'] = f"Based on your documents, here's the relevant information I found:\n\n{context[:800]}..."
                else:
                    result['answer'] = "I couldn't find relevant information in your uploaded documents to answer this question. Please try rephrasing your question or check if your documents contain information about this topic."
            
            # Add to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': question,
                'timestamp': datetime.now().isoformat()
            })
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': result['answer'],
                'sources': result.get('sources', source_refs or []),
                'success': result.get('success', True),
                'timestamp': datetime.now().isoformat()
            })
            
            # Update session state
            st.session_state.question_count += 1
            
            print(f"[v0] Successfully added to chat history. Total messages: {len(st.session_state.chat_history)}")
            
            # Rerun to display new messages
            st.rerun()
            
        except Exception as e:
            print(f"[v0] Error in handle_question: {str(e)}")
            st.error(f"❌ Error generating answer: {str(e)}")
            
            st.session_state.chat_history.append({
                'role': 'user',
                'content': question,
                'timestamp': datetime.now().isoformat()
            })
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': f"I encountered an error while processing your question: {str(e)}\n\nPlease try again or rephrase your question.",
                'sources': [],
                'success': False,
                'timestamp': datetime.now().isoformat()
            })
            
            st.session_state.question_count += 1
            st.rerun()

def handle_image_generation(prompt: str):
    """Handle image generation request"""
    with st.spinner("Generating image..."):
        try:
            success, image_b64, message = st.session_state.image_generator.generate_image(prompt)
            
            # Add to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': f"Generate image: {prompt}",
                'timestamp': datetime.now().isoformat()
            })
            
            if success and image_b64:
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': message,
                    'image': image_b64,
                    'timestamp': datetime.now().isoformat()
                })
                st.success("✅ Image generated successfully!")
            else:
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"I created a placeholder image for '{prompt}'. For AI-generated images, set up a Hugging Face API key.",
                    'image': image_b64 if image_b64 else None,
                    'timestamp': datetime.now().isoformat()
                })
                st.info("💡 For AI-generated images, set HUGGINGFACE_API_KEY environment variable")
            
            st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error generating image: {str(e)}")

def regenerate_last_answer():
    """Regenerate the last assistant answer"""
    if len(st.session_state.chat_history) >= 2:
        # Get the last user question
        last_user_message = None
        for message in reversed(st.session_state.chat_history):
            if message['role'] == 'user':
                last_user_message = message
                break
        
        if last_user_message:
            # Remove the last assistant response
            st.session_state.chat_history = [
                msg for msg in st.session_state.chat_history 
                if not (msg['role'] == 'assistant' and msg == st.session_state.chat_history[-1])
            ]
            
            # Regenerate answer
            handle_question(last_user_message['content'])

def export_chat_history():
    """Export chat history as JSON"""
    if st.session_state.chat_history:
        export_data = {
            'session_id': st.session_state.chat_session_id,
            'export_timestamp': datetime.now().isoformat(),
            'document_stats': st.session_state.search_system.get_statistics() if st.session_state.documents_processed else {},
            'chat_history': st.session_state.chat_history,
            'question_count': st.session_state.question_count
        }
        
        # Create download button
        json_string = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Download Chat History",
            data=json_string,
            file_name=f"studymate_chat_{st.session_state.chat_session_id}.json",
            mime="application/json"
        )
        st.success("✅ Chat history ready for download!")

def export_chat_as_pdf():
    """Export chat history as PDF"""
    if st.session_state.chat_history:
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph("StudyMate Chat History", title_style))
            story.append(Spacer(1, 12))
            
            # Chat messages
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    user_style = ParagraphStyle(
                        'UserMessage',
                        parent=styles['Normal'],
                        fontSize=12,
                        textColor='blue',
                        spaceAfter=6
                    )
                    story.append(Paragraph(f"<b>You:</b> {message['content']}", user_style))
                else:
                    assistant_style = ParagraphStyle(
                        'AssistantMessage',
                        parent=styles['Normal'],
                        fontSize=12,
                        textColor='black',
                        spaceAfter=12
                    )
                    story.append(Paragraph(f"<b>StudyMate:</b> {message['content']}", assistant_style))
                
                story.append(Spacer(1, 6))
            
            doc.build(story)
            buffer.seek(0)
            
            # Create download button
            st.download_button(
                label="💾 Download PDF",
                data=buffer.getvalue(),
                file_name=f"studymate_chat_{st.session_state.chat_session_id}.pdf",
                mime="application/pdf"
            )
            st.success("✅ PDF ready for download!")
            
        except Exception as e:
            st.error(f"❌ Error creating PDF: {str(e)}")

def clear_chat_history():
    """Clear chat history and reset session"""
    st.session_state.chat_history = []
    st.session_state.question_count = 0
    st.session_state.chat_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.success("🗑️ Chat history cleared!")
    st.rerun()

def main():
    """Main application function"""
    initialize_session_state()
    display_header()
    
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_chat_interface()
    
    with col2:
        display_sidebar()

if __name__ == "__main__":
    main()
