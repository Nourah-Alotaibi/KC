"""
RAG-lite functionality for Aafiya AI
Handles document upload and knowledge base integration
"""

import streamlit as st
import PyPDF2
from typing import List, Dict, Any
import re
from io import StringIO
import pandas as pd
import csv

def process_text_file(uploaded_file) -> str:
    """Process uploaded text file"""
    try:
        content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        return content
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return ""

def process_pdf_file(uploaded_file) -> str:
    """Process uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return ""

def process_csv_file(uploaded_file) -> str:
    """Process uploaded CSV file and convert to readable text format"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Convert DataFrame to a readable text format
        content = f"CSV Data from {uploaded_file.name}:\n\n"
        
        # Add column information
        content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        
        # Add data summary
        content += f"Total rows: {len(df)}\n\n"
        
        # Convert each row to readable text
        content += "Data entries:\n"
        for index, row in df.iterrows():
            row_text = f"Row {index + 1}: "
            row_items = []
            for col, value in row.items():
                if pd.notna(value):  # Only include non-null values
                    row_items.append(f"{col}: {value}")
            content += row_text + ", ".join(row_items) + "\n"
            
            # Limit to first 100 rows to avoid too much content
            if index >= 99:
                content += f"... and {len(df) - 100} more rows\n"
                break
        
        return content
        
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better retrieval"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def simple_keyword_search(query: str, chunks: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    """Simple keyword-based search through text chunks"""
    query_words = query.lower().split()
    scored_chunks = []
    
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = 0
        
        # Score based on keyword matches
        for word in query_words:
            if word in chunk_lower:
                score += chunk_lower.count(word)
        
        # Bonus for exact phrase matches
        if query.lower() in chunk_lower:
            score += 5
        
        if score > 0:
            scored_chunks.append({
                'chunk': chunk,
                'score': score,
                'index': i
            })
    
    # Sort by score and return top_k
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    return scored_chunks[:top_k]

def build_context_from_documents(query: str, documents: List[Dict]) -> str:
    """Build context from uploaded nutrition documents"""
    if not documents:
        return ""
    
    all_relevant_chunks = []
    
    for doc in documents:
        chunks = doc.get('chunks', [])
        if chunks:
            relevant = simple_keyword_search(query, chunks, top_k=2)
            for item in relevant:
                all_relevant_chunks.append({
                    'text': item['chunk'],
                    'document': doc['name'],
                    'score': item['score']
                })
    
    if not all_relevant_chunks:
        return ""
    
    # Sort all chunks by score and take top 3
    all_relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
    top_chunks = all_relevant_chunks[:3]
    
    context_parts = []
    for chunk in top_chunks:
        context_parts.append(f"From {chunk['document']}:\n{chunk['text']}")
    
    context = "\n\n--- RELEVANT NUTRITION INFORMATION ---\n"
    context += "\n\n".join(context_parts)
    context += "\n--- END NUTRITION INFORMATION ---\n\n"
    
    return context

def nutrition_document_uploader():
    """Streamlit component for nutrition document upload"""
    
    uploaded_files = st.file_uploader(
        "Upload your documents here:",
        type=['txt', 'pdf', 'csv'],
        accept_multiple_files=True,
        help="Upload nutrition documents, InBody results, medical history, CSV data, etc."
    )
    
    if uploaded_files:
        if 'nutrition_documents' not in st.session_state:
            st.session_state.nutrition_documents = []
        
        # Process new files
        for file in uploaded_files:
            # Check if file already processed
            if not any(doc['name'] == file.name for doc in st.session_state.nutrition_documents):
                with st.spinner(f"Processing {file.name}..."):
                    if file.type == "application/pdf":
                        content = process_pdf_file(file)
                    elif file.type == "text/csv" or file.name.endswith('.csv'):
                        content = process_csv_file(file)
                    else:
                        content = process_text_file(file)
                    
                    if content:
                        chunks = chunk_text(content)
                        
                        document = {
                            'name': file.name,
                            'content': content,
                            'chunks': chunks,
                            'type': file.type,
                            'size': len(content)
                        }
                        
                        st.session_state.nutrition_documents.append(document)
        
        # Display loaded documents
        st.success(f"📄 {len(st.session_state.nutrition_documents)} documents loaded into Aafiya's knowledge base")
        
        with st.expander("View Loaded Documents"):
            for doc in st.session_state.nutrition_documents:
                st.markdown(f"**📄 {doc['name']}**")
                st.text(f"Type: {doc['type']} | Size: {doc['size']} characters | Chunks: {len(doc['chunks'])}")
                st.text(f"Preview: {doc['content'][:150]}...")
                st.divider()
        
        # Clear documents button
        if st.button("🗑️ Clear All Documents"):
            st.session_state.nutrition_documents = []
            st.rerun()
    
    return st.session_state.get('nutrition_documents', [])

def enhance_prompt_with_rag(original_prompt: str, documents: List[Dict]) -> str:
    """Enhance user prompt with relevant information from documents"""
    if not documents:
        return original_prompt
    
    # Build context from documents
    context = build_context_from_documents(original_prompt, documents)
    
    if not context:
        return original_prompt
    
    enhanced_prompt = f"""
{context}

Based on the above nutrition information from the uploaded documents, please answer this question:

{original_prompt}

If the uploaded documents contain relevant information, please reference it in your response. If the documents don't contain relevant information for this specific question, provide your general nutrition expertise.
"""
    
    return enhanced_prompt

def display_rag_info(documents: List[Dict], query: str):
    """Display information about RAG retrieval"""
    if not documents:
        return
    
    st.info("🔍 Aafiya is searching through your uploaded nutrition documents...")
    
    # Show which documents were searched
    doc_names = [doc['name'] for doc in documents]
    st.caption(f"Searched documents: {', '.join(doc_names)}")
    
    # Show relevant chunks found
    context = build_context_from_documents(query, documents)
    if context:
        with st.expander("📋 Retrieved Information from Documents"):
            st.markdown(context)
    else:
        st.caption("ℹ️ No highly relevant information found in uploaded documents for this query.")