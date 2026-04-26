"""
cypher_generator.py
Utility functions for Cypher query generation and execution.
"""
from typing import List, Dict, Any, Tuple
from llm_service import generate_cypher_query
from neo4j_client import Neo4jClient
import logging
import re

logger = logging.getLogger(__name__)


def extract_search_terms(question: str) -> List[str]:
    """
    Extract key search terms from the question.
    
    Args:
        question: User question
        
    Returns:
        List of search terms
    """
    # Remove common words
    stop_words = {"what", "is", "are", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "how", "why", "when", "where", "who"}
    
    # Split and filter
    terms = [word.lower() for word in question.split() if word.lower() not in stop_words and len(word) > 2]
    return terms


def create_fallback_queries(search_terms: List[str]) -> List[str]:
    """
    Create fallback queries for broader searching.
    Tries multiple strategies to find relevant information.
    
    Args:
        search_terms: List of search terms extracted from question
        
    Returns:
        List of fallback Cypher queries
    """
    fallback_queries = []
    
    if not search_terms:
        return fallback_queries
    
    primary_term = search_terms[0]
    
    # Query 1: Find any entity containing first search term (exact match)
    fallback_queries.append(
        f'MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower("{primary_term}") RETURN n.name LIMIT 25'
    )
    
    # Query 2: Find entities containing search terms with their relationships
    if len(search_terms) >= 2:
        secondary_term = search_terms[1]
        fallback_queries.append(
            f'MATCH (n:Entity)-[r]-(m:Entity) WHERE toLower(n.name) CONTAINS toLower("{primary_term}") OR toLower(m.name) CONTAINS toLower("{secondary_term}") RETURN n.name, type(r) as relationship, m.name LIMIT 25'
        )
    
    # Query 3: Find all relationships connected to primary entity
    fallback_queries.append(
        f'MATCH (n:Entity)-[r]-(m:Entity) WHERE toLower(n.name) CONTAINS toLower("{primary_term}") RETURN n.name, type(r) as relationship_type, m.name LIMIT 25'
    )
    
    # Query 4: Multi-hop search (2-3 levels) to find remote connections
    fallback_queries.append(
        f'MATCH (n:Entity)-[*..3]-(m:Entity) WHERE toLower(n.name) CONTAINS toLower("{primary_term}") RETURN DISTINCT n.name, m.name LIMIT 25'
    )
    
    # Query 5: Search for related variations (e.g., "move" -> "movement", "moves")
    # Try common variations of the search term
    variations = []
    if primary_term.endswith('s'):
        variations.append(primary_term[:-1])  # "moves" -> "move"
    else:
        variations.append(primary_term + 's')  # "move" -> "moves"
    
    if len(search_terms) >= 2:
        secondary_term = search_terms[1]
        if secondary_term.endswith('s'):
            variations.append(secondary_term[:-1])
        else:
            variations.append(secondary_term + 's')
    
    for variation in variations:
        fallback_queries.append(
            f'MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower("{variation}") RETURN n.name, n LIMIT 25'
        )
    
    # Query 6: Find all entities that relate to any search term through any relationship
    fallback_queries.append(
        f'MATCH (n:Entity)-[r]-(m:Entity) WHERE toLower(n.name) CONTAINS toLower("{primary_term}") RETURN DISTINCT m.name, type(r) as rel_type LIMIT 25'
    )
    
    # Query 7: Return sample of all entities to understand data structure
    fallback_queries.append(
        'MATCH (n:Entity) RETURN DISTINCT n.name LIMIT 30'
    )
    
    return fallback_queries


def query_graph(question: str, neo4j_client: Neo4jClient) -> Tuple[List[Dict[str, Any]], str]:
    """
    Convert a question to a Cypher query and execute it.
    Tries primary query first, then falls back to broader searches if needed.
    
    Args:
        question: Natural language question
        neo4j_client: Neo4j client instance
        
    Returns:
        Tuple of (query_results, cypher_query)
    """
    try:
        # Try primary LLM-generated query first
        primary_query = generate_cypher_query(question, "")
        
        if primary_query and primary_query.strip() != "":
            logger.info(f"Generated Cypher: {primary_query}")
            
            try:
                results = neo4j_client.run_query(primary_query)
                
                if results:
                    logger.info(f"Query returned {len(results)} results")
                    return results, primary_query
                else:
                    logger.warning("Primary query returned no results, trying fallback queries...")
            except Exception as e:
                logger.warning(f"Primary query failed: {e}, trying fallback queries...")
        
        # Try fallback queries if primary didn't work
        search_terms = extract_search_terms(question)
        fallback_queries = create_fallback_queries(search_terms)
        
        for fallback_query in fallback_queries:
            try:
                logger.info(f"Trying fallback query: {fallback_query}")
                results = neo4j_client.run_query(fallback_query)
                
                if results:
                    logger.info(f"Fallback query returned {len(results)} results")
                    return results, fallback_query
            except Exception as e:
                logger.debug(f"Fallback query failed: {e}")
                continue
        
        logger.warning("All queries returned no results")
        return [], ""
        
    except Exception as e:
        logger.error(f"Error querying graph: {e}")
        return [], ""


def search_entity(entity_name: str, neo4j_client: Neo4jClient) -> List[Dict[str, Any]]:
    """
    Search for an entity by name.
    
    Args:
        entity_name: Name of entity to search
        neo4j_client: Neo4j client instance
        
    Returns:
        Query results
    """
    query = """
    MATCH (e:Entity {name: $name})
    OPTIONAL MATCH (e)-[r]->(neighbors)
    RETURN e.name as entity, r, neighbors.name as neighbor
    """
    
    try:
        results = neo4j_client.run_query(query, {"name": entity_name})
        logger.info(f"Found {len(results)} results for entity: {entity_name}")
        return results
        
    except Exception as e:
        logger.error(f"Error searching entity: {e}")
        return []


def find_relationships(entity1: str, entity2: str, neo4j_client: Neo4jClient) -> List[Dict[str, Any]]:
    """
    Find relationships between two entities.
    
    Args:
        entity1: First entity name
        entity2: Second entity name
        neo4j_client: Neo4j client instance
        
    Returns:
        Paths between the entities
    """
    query = """
    MATCH path = (a:Entity {name: $entity1})-[*..3]->(b:Entity {name: $entity2})
    RETURN path
    LIMIT 5
    """
    
    try:
        results = neo4j_client.run_query(query, {"entity1": entity1, "entity2": entity2})
        logger.info(f"Found {len(results)} paths between {entity1} and {entity2}")
        return results
        
    except Exception as e:
        logger.error(f"Error finding relationships: {e}")
        return []


def get_related_entities(entity_name: str, neo4j_client: Neo4jClient, depth: int = 1) -> List[Dict[str, Any]]:
    """
    Get all entities related to a given entity.
    
    Args:
        entity_name: Entity name
        neo4j_client: Neo4j client instance
        depth: Relationship depth
        
    Returns:
        List of related entities
    """
    query = f"""
    MATCH (e:Entity {{name: $name}})-[*1..{depth}]-(related:Entity)
    RETURN DISTINCT related.name as entity
    """
    
    try:
        results = neo4j_client.run_query(query, {"name": entity_name})
        logger.info(f"Found {len(results)} related entities for: {entity_name}")
        return results
        
    except Exception as e:
        logger.error(f"Error getting related entities: {e}")
        return []
