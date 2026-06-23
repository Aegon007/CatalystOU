"""
GraphRAG Integration Layer for CatalystOU

This module provides an abstraction layer for integrating GraphRAG as an optional
backend for profile extraction and collaboration matching. It maintains compatibility
with the existing LLM provider architecture while enabling graph-based reasoning.

Design Philosophy:
- GraphRAG is pluggable, not mandatory
- Can be used alongside or instead of semantic similarity matching
- Enables knowledge graph-based collaboration discovery
- Gracefully degrades to standard matching if GraphRAG is unavailable
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

try:
    from graphrag.index.graph_rag_index import GraphRAGIndex
    from graphrag.query.query_engine import QueryEngine
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False

from utils.logger import setup_logger

logger = setup_logger("graphrag_integration")


@dataclass
class GraphEntity:
    """Represents an entity in the researcher knowledge graph."""
    name: str
    entity_type: str  # "researcher", "method", "dataset", "application", "domain"
    description: str
    attributes: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.entity_type,
            "description": self.description,
            "attributes": self.attributes or {}
        }


@dataclass
class GraphRelationship:
    """Represents a relationship between entities in the researcher knowledge graph."""
    source: str
    target: str
    relation_type: str  # "uses", "studies", "applies_to", "complements", etc.
    strength: float  # 0.0 to 1.0
    attributes: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type,
            "strength": self.strength,
            "attributes": self.attributes or {}
        }


class ResearcherKnowledgeGraph:
    """
    Manages a knowledge graph representation of researcher profiles.
    Bridges between CatalystOU's profile schema and GraphRAG's entity-relationship model.
    """
    
    def __init__(self, researcher_name: str):
        self.researcher_name = researcher_name
        self.entities: Dict[str, GraphEntity] = {}
        self.relationships: List[GraphRelationship] = []
    
    def add_entity(self, entity: GraphEntity) -> None:
        """Add or update an entity in the graph."""
        key = f"{entity.entity_type}:{entity.name}"
        self.entities[key] = entity
        logger.debug(f"Added entity: {key}")
    
    def add_relationship(self, relationship: GraphRelationship) -> None:
        """Add a relationship between entities."""
        self.relationships.append(relationship)
        logger.debug(f"Added relationship: {relationship.source} -[{relationship.relation_type}]-> {relationship.target}")
    
    def from_researcher_profile(self, profile: Dict[str, Any]) -> None:
        """
        Convert a researcher profile to knowledge graph entities and relationships.
        
        Args:
            profile: ResearcherProfile dict from utils.data.schema
        """
        # Add researcher as central entity
        researcher_entity = GraphEntity(
            name=self.researcher_name,
            entity_type="researcher",
            description=profile.get("Summary Description", ""),
            attributes={
                "affiliation": profile.get("Affiliation", ""),
                "research_focus": profile.get("Research Focus", "")
            }
        )
        self.add_entity(researcher_entity)
        
        # Extract and add domains
        for domain in profile.get("Research Domains", []):
            entity = GraphEntity(
                name=domain,
                entity_type="domain",
                description=f"Research domain studied by {self.researcher_name}"
            )
            self.add_entity(entity)
            self.add_relationship(GraphRelationship(
                source=self.researcher_name,
                target=domain,
                relation_type="studies",
                strength=0.9
            ))
        
        # Extract and add techniques
        for technique in profile.get("Techniques Used", []):
            entity = GraphEntity(
                name=technique,
                entity_type="method",
                description=f"Technique employed by {self.researcher_name}"
            )
            self.add_entity(entity)
            self.add_relationship(GraphRelationship(
                source=self.researcher_name,
                target=technique,
                relation_type="uses",
                strength=0.85
            ))
        
        # Extract and add datasets/platforms
        for data_platform in profile.get("Data & Platforms", []):
            entity = GraphEntity(
                name=data_platform,
                entity_type="dataset",
                description=f"Data/platform utilized by {self.researcher_name}"
            )
            self.add_entity(entity)
            self.add_relationship(GraphRelationship(
                source=self.researcher_name,
                target=data_platform,
                relation_type="uses",
                strength=0.8
            ))
        
        # Extract and add application areas
        for application in profile.get("Application Areas", []):
            entity = GraphEntity(
                name=application,
                entity_type="application",
                description=f"Application area of interest to {self.researcher_name}"
            )
            self.add_entity(entity)
            self.add_relationship(GraphRelationship(
                source=self.researcher_name,
                target=application,
                relation_type="applies_to",
                strength=0.85
            ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph as dictionary for storage or GraphRAG indexing."""
        return {
            "researcher": self.researcher_name,
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "relationships": [rel.to_dict() for rel in self.relationships]
        }
    
    def to_json(self, path: Path) -> None:
        """Save knowledge graph to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved knowledge graph to {path}")


class GraphRAGBackend:
    """
    Wrapper around GraphRAG functionality for CatalystOU.
    Provides profile extraction and collaboration analysis via graph-based reasoning.
    """
    
    def __init__(self, index_dir: Optional[Path] = None):
        """
        Initialize GraphRAG backend.
        
        Args:
            index_dir: Directory to store/load GraphRAG indices
        """
        if not GRAPHRAG_AVAILABLE:
            logger.warning("GraphRAG not installed. Falling back to semantic matching.")
            self.available = False
            return
        
        self.available = True
        self.index_dir = index_dir or Path(".graphrag_indices")
        self.index_dir.mkdir(exist_ok=True)
        self.indices: Dict[str, GraphRAGIndex] = {}
        logger.info("GraphRAG backend initialized")
    
    async def build_researcher_index(self, researcher_name: str, profile: Dict[str, Any]) -> str:
        """
        Build a GraphRAG index for a single researcher profile.
        
        Args:
            researcher_name: Name of researcher
            profile: Researcher profile dict
            
        Returns:
            Index ID for later queries
        """
        if not self.available:
            logger.warning("GraphRAG not available, skipping index build")
            return ""
        
        try:
            kg = ResearcherKnowledgeGraph(researcher_name)
            kg.from_researcher_profile(profile)
            
            index_id = f"{researcher_name.replace(' ', '_')}"
            index_path = self.index_dir / index_id
            index_path.mkdir(exist_ok=True)
            
            # Save graph representation
            kg.to_json(index_path / "knowledge_graph.json")
            
            # In production: initialize actual GraphRAGIndex
            # For now: store graph representation
            self.indices[index_id] = kg
            
            logger.info(f"Built index for {researcher_name}: {len(kg.entities)} entities, {len(kg.relationships)} relationships")
            return index_id
        except Exception as e:
            logger.error(f"Error building GraphRAG index: {e}")
            return ""
    
    async def query_collaboration_synergies(
        self, 
        index_a: str, 
        index_b: str,
        query: str = "What are the collaboration opportunities between these two researchers?"
    ) -> Dict[str, Any]:
        """
        Query GraphRAG indices for collaboration synergies.
        
        Args:
            index_a: First researcher's index ID
            index_b: Second researcher's index ID
            query: Natural language query
            
        Returns:
            Collaboration analysis with reasoning paths
        """
        if not self.available:
            logger.warning("GraphRAG not available, cannot perform graph query")
            return {}
        
        try:
            kg_a = self.indices.get(index_a)
            kg_b = self.indices.get(index_b)
            
            if not kg_a or not kg_b:
                logger.error(f"One or both indices not found: {index_a}, {index_b}")
                return {}
            
            # Analyze common entities and paths
            entities_a = set(e.split(":")[-1] for e in kg_a.entities.keys())
            entities_b = set(e.split(":")[-1] for e in kg_b.entities.keys())
            
            common_entities = entities_a & entities_b
            complementary_entities = (entities_a ^ entities_b)
            
            analysis = {
                "query": query,
                "common_entities": list(common_entities),
                "complementary_entities": list(complementary_entities),
                "researcher_a": kg_a.researcher_name,
                "researcher_b": kg_b.researcher_name,
                "graph_entities_a": len(kg_a.entities),
                "graph_entities_b": len(kg_b.entities),
                "graph_relationships_a": len(kg_a.relationships),
                "graph_relationships_b": len(kg_b.relationships),
            }
            
            logger.info(f"Graph query completed: {len(common_entities)} common entities found")
            return analysis
        except Exception as e:
            logger.error(f"Error querying GraphRAG indices: {e}")
            return {}
    
    def get_index_path(self, index_id: str) -> Optional[Path]:
        """Get filesystem path to an index."""
        path = self.index_dir / index_id
        return path if path.exists() else None


class HybridCollaborationAnalyzer:
    """
    Combines semantic similarity matching with GraphRAG-based reasoning.
    Provides both shallow (similarity) and deep (graph) analysis.
    """
    
    def __init__(self, use_graphrag: bool = False):
        self.graphrag_backend = GraphRAGBackend() if use_graphrag else None
        self.graphrag_enabled = use_graphrag and (self.graphrag_backend.available if self.graphrag_backend else False)
    
    async def analyze_collaboration(
        self,
        profile_a: Dict[str, Any],
        profile_b: Dict[str, Any],
        semantic_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze collaboration using both semantic and graph-based methods.
        
        Args:
            profile_a: First researcher profile
            profile_b: Second researcher profile
            semantic_scores: Pre-computed semantic similarity scores
            
        Returns:
            Hybrid analysis combining both approaches
        """
        analysis = {
            "semantic_analysis": semantic_scores,
            "graph_analysis": None
        }
        
        if self.graphrag_enabled and self.graphrag_backend:
            try:
                # Build indices
                index_a = await self.graphrag_backend.build_researcher_index(
                    profile_a.get("Researcher Profile", "A"),
                    profile_a
                )
                index_b = await self.graphrag_backend.build_researcher_index(
                    profile_b.get("Researcher Profile", "B"),
                    profile_b
                )
                
                # Query for synergies
                if index_a and index_b:
                    graph_analysis = await self.graphrag_backend.query_collaboration_synergies(
                        index_a, index_b
                    )
                    analysis["graph_analysis"] = graph_analysis
            except Exception as e:
                logger.error(f"GraphRAG analysis failed: {e}, falling back to semantic only")
        
        return analysis


def get_graphrag_status() -> Dict[str, Any]:
    """Get status of GraphRAG availability and configuration."""
    return {
        "available": GRAPHRAG_AVAILABLE,
        "installed": GRAPHRAG_AVAILABLE,
        "message": "GraphRAG is available" if GRAPHRAG_AVAILABLE else "GraphRAG not installed. Install with: pip install graphrag"
    }
