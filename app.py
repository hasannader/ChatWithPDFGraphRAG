"""
Graph RAG PDF Chatbot: A Retrieval-Augmented Generation system that extracts knowledge graphs from PDFs using Azure OpenAI LLM.
Stores entities and relationships in Neo4j, enabling intelligent question-answering grounded in document content with Cypher query generation and fallback search strategies for accurate, context-aware responses.

app.py
Main Streamlit application for the Graph RAG PDF Chatbot.
"""
import streamlit as st
import tempfile
import os
from datetime import datetime

from pdf_processor import process_pdf
from graph_builder import build_graph_from_pdf, get_graph_stats
from neo4j_client import Neo4jClient
from rag_engine import RAGEngine
from memory import ConversationMemory
from config import MAX_PDF_SIZE_MB

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Graph RAG PDF Chatbot",
    page_icon="📄🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄🕸️ Graph RAG PDF Chatbot")
st.markdown("Chat with your PDFs using Neo4j Graph Database and AI")

# Initialize session state
if "neo4j_client" not in st.session_state:
    try:
        st.session_state.neo4j_client = Neo4jClient()
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        st.stop()

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine(st.session_state.neo4j_client)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph_built" not in st.session_state:
    st.session_state.graph_built = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    menu = st.radio(
        "Navigation",
        ["💬 Chat", "📤 Upload PDF", "📊 Graph Info", "🔄 Reset"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", help="Clear all messages in current session"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.success("Chat history cleared!")
    
    st.divider()
    
    # Graph statistics
    if st.session_state.graph_built:
        st.subheader("📈 Graph Statistics")
        try:
            stats = get_graph_stats(st.session_state.neo4j_client)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📍 Nodes", stats["node_count"])
            with col2:
                st.metric("🔗 Relationships", stats["relationship_count"])
        except Exception as e:
            st.warning(f"Could not fetch stats: {e}")

# Main content
if menu == "💬 Chat":
    st.header("💬 Chat with Your PDF")
    
    if not st.session_state.graph_built:
        st.info("📤 Please upload a PDF first to start chatting!")
    else:
        st.success(f"✅ PDF loaded: {st.session_state.pdf_name}")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "metadata" in message and message["metadata"]:
                    with st.expander("📊 Details"):
                        st.json(message["metadata"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.memory.add_message("user", prompt)
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process with RAG engine
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Get conversation context
                        context = st.session_state.memory.get_context(num_messages=4)
                        
                        # Process question
                        result = st.session_state.rag_engine.process_question(
                            prompt,
                            context
                        )
                        
                        # Display answer
                        st.markdown(result["answer"])
                        
                        # Store message with metadata
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "metadata": {
                                "cypher_query": result["cypher_query"],
                                "results_count": len(result["graph_results"])
                            }
                        })
                        
                        # Add to memory
                        st.session_state.memory.add_message(
                            "assistant",
                            result["answer"],
                            {"results_count": len(result["graph_results"])}
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing question: {e}")
                        logger.error(f"Error: {e}")


elif menu == "📤 Upload PDF":
    st.header("📤 Upload PDF")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        if file_size_mb > MAX_PDF_SIZE_MB:
            st.error(f"File is too large. Maximum size: {MAX_PDF_SIZE_MB}MB, Your file: {file_size_mb:.2f}MB")
        else:
            st.success(f"✅ File ready: {uploaded_file.name} ({file_size_mb:.2f}MB)")
            
            if st.button("🚀 Process and Build Graph", type="primary"):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process PDF
                    with st.spinner("📖 Processing PDF..."):
                        chunks = process_pdf(tmp_path)
                        st.info(f"📚 Extracted {len(chunks)} text chunks")
                    
                    # Build graph
                    with st.spinner("🕸️ Building knowledge graph..."):
                        build_graph_from_pdf(chunks, st.session_state.neo4j_client)
                    
                    # Update session state
                    st.session_state.graph_built = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []
                    st.session_state.memory.clear()
                    
                    # Show stats
                    stats = get_graph_stats(st.session_state.neo4j_client)
                    st.success("✅ Graph built successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📍 Nodes Created", stats["node_count"])
                    with col2:
                        st.metric("🔗 Relationships Created", stats["relationship_count"])
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                    logger.error(f"Error: {e}")


elif menu == "📊 Graph Info":
    st.header("📊 Graph Information")
    
    if not st.session_state.graph_built:
        st.info("No graph built yet. Please upload a PDF first.")
    else:
        try:
            stats = get_graph_stats(st.session_state.neo4j_client)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📍 Total Nodes", stats["node_count"])
            with col2:
                st.metric("🔗 Total Relationships", stats["relationship_count"])
            
            st.divider()
            
            # Sample entities
            st.subheader("Sample Entities")
            query = "MATCH (n:Entity) RETURN n.name as name LIMIT 10"
            results = st.session_state.neo4j_client.run_query(query)
            if results:
                entities = [r["name"] for r in results]
                st.write("\n".join([f"• {e}" for e in entities]))
            else:
                st.info("No entities found in the graph.")
            
        except Exception as e:
            st.error(f"Error retrieving graph info: {e}")


elif menu == "🔄 Reset":
    st.header("🔄 Reset System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗑️ Clear Database")
        st.warning("⚠️ This will delete all nodes and relationships from the graph database!")
        
        if st.button("Clear Graph Database", type="secondary", key="clear_db"):
            try:
                with st.spinner("Clearing database..."):
                    st.session_state.neo4j_client.clear_database()
                
                st.session_state.graph_built = False
                st.session_state.pdf_name = None
                
                st.success("✅ Graph database cleared successfully!")
                
            except Exception as e:
                st.error(f"Error clearing database: {e}")
                logger.error(f"Error: {e}")
    
    with col2:
        st.subheader("🔄 Full Reset")
        st.warning("⚠️ This will clear the database and chat history!")
        
        if st.button("Clear Everything", type="secondary", key="clear_all"):
            try:
                with st.spinner("Clearing database and chat..."):
                    st.session_state.neo4j_client.clear_database()
                
                st.session_state.graph_built = False
                st.session_state.pdf_name = None
                st.session_state.messages = []
                st.session_state.memory.clear()
                
                st.success("✅ System reset successfully!")
                
            except Exception as e:
                st.error(f"Error resetting system: {e}")
                logger.error(f"Error: {e}")
