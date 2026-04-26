"""
neo4j_client.py
Neo4j database connection and query execution.
"""
from neo4j import GraphDatabase, Session
from typing import List, Dict, Any
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
import logging

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Client for interacting with Neo4j database."""
    
    def __init__(self):
        """Initialize Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
                encrypted=False
            )
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    def run_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        if params is None:
            params = {}
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                records = [record.data() for record in result]
                logger.info(f"Query executed successfully, returned {len(records)} records")
                return records
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def create_entity(self, label: str, properties: Dict[str, Any]) -> None:
        """
        Create or merge a node with specified properties.
        
        Args:
            label: Node label
            properties: Node properties
        """
        prop_keys = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        query = f"MERGE (n:{label} {{{prop_keys}}})"
        
        try:
            with self.driver.session() as session:
                session.run(query, properties)
                logger.info(f"Created/merged entity: {label}")
        except Exception as e:
            logger.error(f"Error creating entity: {e}")
            raise
    
    def create_relationship(self, entity1: str, rel_type: str, entity2: str, rel_properties: Dict = None) -> None:
        """
        Create a relationship between two entities.
        
        Args:
            entity1: First entity name
            rel_type: Relationship type
            entity2: Second entity name
            rel_properties: Relationship properties
        """
        if rel_properties is None:
            rel_properties = {}
        
        prop_str = ""
        if rel_properties:
            prop_keys = ", ".join([f"{k}: ${k}" for k in rel_properties.keys()])
            prop_str = f" {{{prop_keys}}}"
        
        query = f"""
        MATCH (a {{name: $entity1}})
        MATCH (b {{name: $entity2}})
        MERGE (a)-[r:{rel_type}{prop_str}]->(b)
        """
        
        params = {
            "entity1": entity1,
            "entity2": entity2,
            **rel_properties
        }
        
        try:
            with self.driver.session() as session:
                session.run(query, params)
                logger.info(f"Created relationship: {entity1} -[{rel_type}]-> {entity2}")
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            raise
    
    def create_constraint(self, label: str, property_name: str) -> None:
        """
        Create a unique constraint on a property.
        
        Args:
            label: Node label
            property_name: Property name
        """
        query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
        
        try:
            with self.driver.session() as session:
                session.run(query)
                logger.info(f"Created constraint on {label}.{property_name}")
        except Exception as e:
            logger.error(f"Error creating constraint: {e}")
    
    def clear_database(self) -> None:
        """Delete all nodes and relationships in the database."""
        query = "MATCH (n) DETACH DELETE n"
        
        try:
            with self.driver.session() as session:
                session.run(query)
                logger.info("Database cleared")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
