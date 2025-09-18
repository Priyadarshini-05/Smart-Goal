import streamlit as st
from typing import List, Dict

def display_sources(sources: List[Dict]):
    """Display source references in a clean, formatted way"""
    if sources:
        st.markdown("**📚 Sources:**")
        source_list = []
        for source in sources[:3]:  # Limit to 3 sources for cleaner display
            filename = source.get('filename', 'Unknown')
            page_num = source.get('page_num', 'Unknown')
            source_list.append(f"• {filename}, Page {page_num}")
        
        st.markdown(" • ".join([f"**{filename}, Page {page_num}**" for source in sources[:3] 
                               for filename, page_num in [(source.get('filename', 'Unknown'), 
                                                          source.get('page_num', 'Unknown'))]]))
