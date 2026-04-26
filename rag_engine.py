"""
rag_engine.py
RAG (Retrieval Augmented Generation) engine that orchestrates the complete flow.
"""
from typing import List, Dict, Tuple
from cypher_generator import query_graph
from llm_service import generate_answer
from neo4j_client import Neo4jClient
import logging

logger = logging.getLogger(__name__)


class RAGEngine:
    """Main RAG engine for answering questions based on graph knowledge."""
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize RAG engine.
        
        Args:
            neo4j_client: Neo4j client instance
        """
        self.neo4j_client = neo4j_client
    
    def retrieve_context(self, question: str) -> Tuple[List[Dict], str]:
        """
        Retrieve context from the graph based on the question.
        
        Args:
            question: User question
            
        Returns:
            Tuple of (query_results, cypher_query)
        """
        try:
            # Query the graph and get both results and cypher query
            results, cypher_query = query_graph(question, self.neo4j_client)
            
            logger.info(f"Retrieved {len(results)} results from graph")
            logger.info(f"Cypher Query: {cypher_query}")
            return results, cypher_query
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return [], ""
    
    def process_question(self, question: str, chat_history: List[Dict] = None) -> Dict:
        """
        Process a question and generate an answer using RAG.
        
        Args:
            question: User question
            chat_history: Previous conversation history
            
        Returns:
            Dictionary containing results and answer
        """
        if chat_history is None:
            chat_history = []
        
        try:
            # Step 1: Retrieve context from graph
            graph_results, cypher_query = self.retrieve_context(question)
            
            logger.info(f"Cypher Query captured: {cypher_query}")
            logger.info(f"Graph Results: {graph_results}")
            
            # Step 2: Format results
            formatted_results = self._format_results(graph_results)
            
            # Step 3: Generate answer using LLM with actual results
            answer = generate_answer(question, graph_results, chat_history)
            
            logger.info("Successfully processed question and generated answer")
            
            return {
                "question": question,
                "cypher_query": cypher_query,
                "graph_results": graph_results,
                "formatted_results": formatted_results,
                "answer": answer
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "cypher_query": "",
                "graph_results": [],
                "formatted_results": [],
                "answer": f"Error: {str(e)}"
            }
    
    def _format_results(self, results: List[Dict]) -> List[str]:
        """
        Format raw Neo4j results into readable strings.
        
        Args:
            results: Raw results from Neo4j
            
        Returns:
            List of formatted result strings
        """
        formatted = []
        for result in results:
            formatted_result = ", ".join([f"{k}: {v}" for k, v in result.items()])
            formatted.append(formatted_result)
        
        return formatted
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            stats_query = """
            MATCH (n:Entity)
            RETURN count(n) as node_count,
                   count{MATCH ()-[]->()}  as relationship_count
            """
            results = self.neo4j_client.run_query(stats_query)
            
            if results:
                return {
                    "success": True,
                    "stats": results[0]
                }
            else:
                return {
                    "success": False,
                    "stats": {}
                }
                
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {
                "success": False,
                "stats": {},
                "error": str(e)
            }
