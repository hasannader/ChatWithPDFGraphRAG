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
    Generate a valid Cypher query to answer the user's question.
    
    Graph Structure:
    - Nodes are labeled 'Entity' with properties: name, and other attributes
    - Relationships can be of various types (e.g., RELATED_TO, MENTIONS, DESCRIBES, IS_A, WORKS_AT, LEADS, etc.)
    - ALL entities are connected through various relationship types
    
    IMPORTANT RULES FOR FLEXIBLE QUERIES:
    1. Return ONLY the Cypher query, no explanation
    2. Do NOT include markdown code blocks or backticks
    3. Use CONTAINS operator for fuzzy matching (case-insensitive): WHERE n.name CONTAINS $search
    4. Avoid exact entity name matching - be flexible and exploratory
    5. Return multiple paths and relationships to find relevant information
    6. Use optional MATCH for relationships that might not always exist
    7. Return entity names, relationships, and related entities
    8. Always use LIMIT to keep results manageable (LIMIT 20)
    9. Prioritize broader search patterns over narrow ones
    
    Example queries to follow:
    - MATCH (n:Entity) WHERE n.name CONTAINS $search RETURN n.name, n LIMIT 20
    - MATCH (n:Entity)-[r]-(m:Entity) WHERE n.name CONTAINS $search OR m.name CONTAINS $search RETURN n.name, r, m.name LIMIT 20
    - MATCH (n:Entity) WHERE n.name CONTAINS $search OPTIONAL MATCH (n)-[r]->(m) RETURN n.name, r, m.name LIMIT 20
    - MATCH (n:Entity)-[*..2]-(m:Entity) WHERE n.name CONTAINS $search OR m.name CONTAINS $search RETURN n.name, m.name LIMIT 20
    
    QUERY GENERATION TIPS:
    - Extract key search terms from the question (e.g., "Pawn Moves" -> search for "Pawn" and "Moves")
    - Use CONTAINS with lowercase conversion for flexible matching
    - Mix of MATCH and OPTIONAL MATCH to find different relationship patterns
    - Return both short paths (1 hop) and longer paths (2-3 hops) to ensure you find relevant information
    
    Return a single Cypher query that is most likely to find relevant results.
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
    Based on the user's question and actual database results from the Neo4j query.
    
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
    You are a helpful AI assistant that answers questions based on provided database results.
    Use ONLY the provided database results to answer the user's question.
    
    IMPORTANT RULES:
    1. Answer ONLY based on the database results provided
    2. If results are empty or very limited, clearly state what information IS available
    3. Be factual and accurate - do NOT make up information
    4. If results don't directly answer the question, explain what information was found instead
    5. Provide concise, clear answers
    6. If you find related but not exact matches, mention them
    7. Format the answer to be helpful and highlight what was found in the data
    
    RESPONSE GUIDELINES:
    - If results are found: Answer based on the data, be specific
    - If no results found: Suggest what was searched for and acknowledge the lack of exact matches
    - Always be factual to what's in the database
    """
    
    # Format graph results for the prompt
    results_str = ""
    if graph_results:
        results_str = "Database Results:\n"
        for i, result in enumerate(graph_results, 1):
            results_str += f"{i}. {result}\n"
    else:
        results_str = "Database Results: No results found for this query.\n"
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add relevant chat history for context
    if chat_history:
        for msg in chat_history[-3:]:  # Keep last 3 messages for context
            if msg.get("role") in ["user", "assistant"]:
                messages.append(msg)
    
    # Add current question with database results
    user_message = f"{results_str}\n\nUser Question: {question}\n\nBased on the database results above, please answer the user's question. If no results were found, explain what was searched for and suggest related information if available."
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.3,  # Lower temperature for more factual answers
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info("Successfully generated answer based on database results")
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
