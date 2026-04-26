"""
llm_service.py
Interactions with Azure OpenAI for entity extraction, query generation, and answer synthesis.
"""
from openai import AzureOpenAI
from typing import List, Dict
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME
)
import logging

logger = logging.getLogger(__name__)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)


def extract_entities_and_relationships(text: str) -> str:
    """
    Use LLM to extract entities and relationships from text.
    
    Args:
        text: Input text to extract entities from
        
    Returns:
        Extracted relationships in format: (Entity1)-[RELATION]->(Entity2)
    """
    system_prompt = """
    You are an expert at extracting structured knowledge from text.
    Extract entities (people, places, concepts, objects) and their relationships from the given text.
    
    Format your response as a series of relationships in this exact format:
    (Entity1)-[RELATIONSHIP_TYPE]->(Entity2)
    
    Each relationship on a new line.
    Do NOT include any other text or explanations.
    If no relationships exist, return an empty response.
    """
    
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content.strip()
        logger.info("Successfully extracted entities and relationships")
        return result
        
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return ""


def generate_cypher_query(question: str, graph_schema: str) -> str:
    """
    Generate a Cypher query from a natural language question.
    
    Args:
        question: User question
        graph_schema: Description of the graph structure
        
    Returns:
        Cypher query string
    """
    system_prompt = """
    You are an expert Neo4j Cypher query writer.
    Generate a valid Cypher query to find information relevant to answering the user's question.
    
    Graph Structure:
    - Nodes are labeled 'Entity' with properties: name, and other attributes
    - Relationships can be of various types (e.g., RELATED_TO, MENTIONS, DESCRIBES, IS_A, WORKS_AT, LEADS, ESCAPES_CHECK_BY, CAPTURES, etc.)
    - ALL entities are connected through various relationship types
    
    CRITICAL REQUIREMENTS:
    1. Return ONLY the Cypher query, no explanation
    2. Do NOT include markdown code blocks or backticks
    3. Search for:
       - Exact entities matching key terms
       - Related entities connected by relationships
       - Conceptual matches (e.g., "moves" -> find "move", "movement", "motion", "acts")
       - Entities that are objects of actions (e.g., "king moves" -> find "King" and its relationships)
    4. Use CONTAINS operator for flexible matching (case-insensitive)
    5. Return multiple relationship paths to capture full context
    6. Use LIMIT 25 to keep results manageable but comprehensive
    7. Prioritize finding connected entities and their relationships
    
    Example queries for different question types:
    - For "what is X": MATCH (n:Entity)-[r]->(m:Entity) WHERE toLower(n.name) CONTAINS toLower("X") RETURN n.name, type(r), m.name LIMIT 25
    - For "how does X Y": MATCH (n:Entity)-[r]-(m:Entity) WHERE toLower(n.name) CONTAINS toLower("X") OR toLower(m.name) CONTAINS toLower("Y") RETURN n.name, type(r), m.name, m LIMIT 25
    - For "X moves": MATCH (n:Entity)-[r]-(m:Entity) WHERE toLower(n.name) CONTAINS toLower("X") RETURN n.name, type(r) as action, m.name LIMIT 25
    - For concepts: MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower("keyword") OPTIONAL MATCH (n)-[r]-(m:Entity) RETURN n.name, type(r), m.name LIMIT 25
    
    Return a single comprehensive Cypher query that will find the most relevant information.
    """
    
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}"}
            ],
            temperature=0.1,  # Lower temp for more consistent queries
            max_tokens=600
        )
        
        query = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if query.startswith("```"):
            query = "\n".join(query.split("\n")[1:-1])
        
        logger.info(f"Generated Cypher query: {query}")
        return query
        
    except Exception as e:
        logger.error(f"Error generating Cypher query: {e}")
        return ""


def generate_answer(question: str, graph_results: List[Dict], chat_history: List[Dict] = None) -> str:
    """
    Generate a natural language answer using graph results and chat history.
    Based STRICTLY on the user's question and actual database results from the Neo4j query.
    
    Args:
        question: User question
        graph_results: Results from Neo4j query (actual database results)
        chat_history: Previous conversation history
        
    Returns:
        Natural language answer
    """
    if chat_history is None:
        chat_history = []
    
    system_prompt = """
    You are an AI assistant that converts database results into natural, user-friendly answers.
    
    ⚠️ CRITICAL RULES - DO NOT VIOLATE:
    1. ONLY use information explicitly present in the database results
    2. Convert database records into natural, readable sentences
    3. NEVER show technical database notation like tuples, relationship names in ALL_CAPS, or brackets
    4. NEVER add general knowledge, common sense, or external information
    5. Present information as flowing prose, not as database tuples
    6. If results are empty - say clearly "No relevant information found in the database"
    7. Group related information together logically
    
    ✅ DO THIS:
    Instead of: ('King', 'MOVES_TO', 'One Square in Any Direction')
    Say: "The King can move one square in any direction."
    
    Instead of: [('Black Piece'), ('White Piece')]
    Say: "The board has Black and White pieces."
    
    ✅ RESPONSE GUIDELINES:
    - Transform database output into natural English
    - Use connecting words: can, does, has, includes, involves, requires
    - Group related concepts together
    - Only mention what's explicitly in the database results
    - If results don't fully answer, acknowledge what IS available
    """
    
    # Format graph results for the prompt - show results in readable format
    results_str = ""
    if graph_results:
        results_str = "DATABASE INFORMATION:\n"
        for i, result in enumerate(graph_results, 1):
            results_str += f"{i}. {result}\n"
    else:
        results_str = "DATABASE INFORMATION: No records found.\n"
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add relevant chat history for context
    if chat_history:
        for msg in chat_history[-3:]:  # Keep last 3 messages for context
            if msg.get("role") in ["user", "assistant"]:
                messages.append(msg)
    
    # Add current question with database results
    user_message = f"{results_str}\nQuestion: {question}\n\nConvert the database information above into a natural, user-friendly answer. Do NOT show technical database notation. Do NOT add external knowledge. Present the information as clear, readable sentences."
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.1,  # Very low temperature to enforce strict adherence
            max_tokens=1200
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info("Successfully generated clean, user-friendly answer from database results")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "I encountered an error while generating the answer."


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        List of embedding vectors
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return []
