"""
graph_builder.py
Handles entity and relationship extraction from text chunks and graph building.
"""
import json
import re
from typing import List, Dict, Tuple
from llm_service import extract_entities_and_relationships
from neo4j_client import Neo4jClient
import logging

logger = logging.getLogger(__name__)


def parse_relationships(text: str) -> List[Tuple[str, str, str]]:
    """
    Parse relationship text in format: (Entity1)-[RELATION]->(Entity2)
    
    Args:
        text: Text containing relationships
        
    Returns:
        List of tuples: (entity1, relation_type, entity2)
    """
    relationships = []
    
    # Pattern to match: (Entity1)-[RELATION]->(Entity2)
    pattern = r'\(([^)]+)\)\s*-\[([^\]]+)\]\s*->\s*\(([^)]+)\)'
    matches = re.findall(pattern, text)
    
    for entity1, relation, entity2 in matches:
        relationships.append((entity1.strip(), relation.strip(), entity2.strip()))
    
    return relationships


def extract_graph_from_chunk(chunk: str, neo4j_client: Neo4jClient) -> None:
    """
    Extract entities and relationships from a text chunk and add to Neo4j.
    
    Args:
        chunk: Text chunk to process
        neo4j_client: Neo4j client instance
    """
    try:
        # Use LLM to extract entities and relationships
        extraction_result = extract_entities_and_relationships(chunk)
        
        if not extraction_result:
            logger.warning("No entities extracted from chunk")
            return
        
        # Parse relationships from LLM output
        relationships = parse_relationships(extraction_result)
        
        # Create nodes and relationships in Neo4j
        processed_entities = set()
        for entity1, relation_type, entity2 in relationships:
            # Create entity nodes
            if entity1 not in processed_entities:
                neo4j_client.create_entity("Entity", {"name": entity1})
                processed_entities.add(entity1)
            
            if entity2 not in processed_entities:
                neo4j_client.create_entity("Entity", {"name": entity2})
                processed_entities.add(entity2)
            
            # Create relationship
            neo4j_client.create_relationship(entity1, relation_type, entity2)
        
        logger.info(f"Extracted and processed {len(relationships)} relationships from chunk")
        
    except Exception as e:
        logger.error(f"Error extracting graph from chunk: {e}")


def build_graph_from_pdf(chunks: List[str], neo4j_client: Neo4jClient) -> None:
    """
    Build complete graph from PDF chunks.
    
    Args:
        chunks: List of text chunks from PDF
        neo4j_client: Neo4j client instance
    """
    try:
        # Create unique constraint
        neo4j_client.create_constraint("Entity", "name")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            extract_graph_from_chunk(chunk, neo4j_client)
        
        logger.info("Graph building completed successfully")
        
    except Exception as e:
        logger.error(f"Error building graph: {e}")
        raise


def get_graph_stats(neo4j_client: Neo4jClient) -> Dict:
    """
    Get statistics about the graph.
    
    Args:
        neo4j_client: Neo4j client instance
        
    Returns:
        Dictionary with graph statistics
    """
    try:
        nodes_query = "MATCH (n) RETURN count(n) as node_count"
        relationships_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
        
        nodes = neo4j_client.run_query(nodes_query)
        relationships = neo4j_client.run_query(relationships_query)
        
        return {
            "node_count": nodes[0].get("node_count", 0) if nodes else 0,
            "relationship_count": relationships[0].get("rel_count", 0) if relationships else 0
        }
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        return {"node_count": 0, "relationship_count": 0}
