# 📄🕸️ Graph RAG PDF Chatbot

A production-ready, conversational AI application that combines PDFs, Neo4j Graph Database, and Azure OpenAI to enable intelligent document-based question answering.

## 🎯 Features

- **PDF Processing**: Upload and process PDF documents
- **Knowledge Graph**: Automatically extracts entities and relationships from PDFs and stores them in Neo4j
- **Conversational AI**: Chat-based interface with conversation memory
- **Graph-Based Retrieval**: Uses Neo4j Cypher queries for context retrieval
- **RAG (Retrieval Augmented Generation)**: Combines graph context with LLM for accurate answers
- **Session Management**: Maintains conversation history per session

## 🧩 Tech Stack

- **Python 3.8+**
- **Streamlit**: Interactive web UI
- **Neo4j**: Graph database
- **Azure OpenAI**: LLM for text generation and entity extraction
- **PyPDF/pdfplumber**: PDF processing
- **FAISS** (optional): Vector embeddings for semantic search

## 📁 Project Structure

```
graph_rag_pdf_chatbot/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration and environment variables
├── pdf_processor.py      # PDF loading and text chunking
├── neo4j_client.py       # Neo4j connection and query execution
├── graph_builder.py      # Entity/relationship extraction and graph building
├── llm_service.py        # Azure OpenAI integration
├── cypher_generator.py   # Cypher query generation utilities
├── rag_engine.py         # RAG orchestration engine
├── memory.py             # Conversation memory management
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🚀 Installation

### 1. Clone or Extract the Repository

```bash
cd graph_rag_pdf_chatbot
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### 5. Ensure Neo4j is Running

Neo4j must be running before starting the application. You can:
- Run Neo4j locally: Download from [neo4j.com](https://neo4j.com/download/)
- Or use a Neo4j instance (Aura, Enterprise, etc.)
- Update `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` in `.env` with your connection details

## 📖 Usage

### Start the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

### Using the Application

1. **Upload PDF**:
   - Go to the "📤 Upload PDF" section
   - Select a PDF file (max 50MB)
   - Click "🚀 Process and Build Graph"
   - Wait for the knowledge graph to be built

2. **Chat**:
   - Switch to the "💬 Chat" section
   - Ask questions about the uploaded PDF
   - View answers grounded in the extracted knowledge

3. **View Graph Info**:
   - Check "📊 Graph Info" to see statistics
   - View sample entities extracted from the PDF

4. **Reset**:
   - Use "🔄 Reset" to clear the database and start fresh

## 🧱 How It Works

### 1. PDF Processing
- Extracts text from uploaded PDF
- Splits text into overlapping chunks

### 2. Entity & Relationship Extraction
- Uses Azure OpenAI to extract entities and relationships
- Formats as: `(Entity1)-[RELATIONSHIP_TYPE]->(Entity2)`

### 3. Graph Building
- Creates nodes for each entity
- Creates relationships in Neo4j
- Enforces unique entity names via constraints

### 4. Query Processing
- User asks a question
- System generates Cypher query using LLM
- Executes query against the graph
- Retrieves relevant context

### 5. Answer Generation
- Combines chat history and graph context
- Uses LLM to generate natural language answer
- Returns grounded, accurate response

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit UI and application orchestration |
| `config.py` | Environment variables and configuration |
| `pdf_processor.py` | PDF extraction and text chunking |
| `neo4j_client.py` | Neo4j connection and CRUD operations |
| `graph_builder.py` | Entity extraction and graph construction |
| `llm_service.py` | Azure OpenAI API integration |
| `cypher_generator.py` | Cypher query generation and execution |
| `rag_engine.py` | RAG pipeline orchestration |
| `memory.py` | Conversation history management |

## ⚙️ Configuration Options

In `config.py`:

- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MAX_PDF_SIZE_MB`: Maximum PDF size (default: 50MB)
- `EMBEDDING_DIM`: Embedding dimension (default: 1536)

## 🧪 Testing

Run individual components:

```bash
# Test PDF processor
python pdf_processor.py

# Test Neo4j connection
python neo4j_client.py

# Test LLM service
python llm_service.py
```

## 🔒 Security Considerations

- Store API keys in `.env`, never commit to version control
- Use environment variables for all credentials
- Neo4j connection uses authentication
- Input validation on all LLM prompts

## 📊 Performance Tips

- For large PDFs, increase `CHUNK_SIZE` for faster processing
- Use FAISS for vector-based semantic search (optional)
- Consider indexing frequently queried properties in Neo4j
- Batch process multiple entities for better performance

## 🐛 Troubleshooting

### Neo4j Connection Failed
- Ensure Neo4j is running and accessible
- Check credentials in `.env`
- Verify URI and port

### Azure OpenAI API Errors
- Verify API key and endpoint
- Check deployment name matches
- Ensure API version is correct

### PDF Processing Issues
- Ensure PDF is not corrupted
- Check file size is under limit
- Try with a simpler PDF first

## 📚 Future Enhancements

- [ ] Vector embeddings with FAISS
- [ ] Graph visualization
- [ ] Batch PDF processing
- [ ] Custom entity types
- [ ] Query result caching
- [ ] Multi-user support
- [ ] Export conversation logs

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Support

For issues or questions, please refer to the documentation or create an issue in the repository.

---

**Happy chatting with your PDFs! 📚🤖**
