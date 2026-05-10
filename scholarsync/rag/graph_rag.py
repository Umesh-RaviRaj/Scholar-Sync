"""
GraphRAG engine — stores entities and relationships in Neo4j for
cross-document reasoning, multi-hop queries, and entity linking.

IMPROVED:
  - Dynamic Neo4j relationship types (USES, IMPROVES_UPON, etc.)
  - Rich graph export with relationship descriptions, entity types, degree
  - Orphan node filtering
  - Weighted edges based on relationship type
"""

from __future__ import annotations

import re
from typing import Any

from neo4j import GraphDatabase

from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import Entity, Relationship
from scholarsync.rag.entity_normalizer import (
    normalize_entity_name,
    deduplicate_entities,
    deduplicate_relationships,
    infer_category,
    build_category_edges,
)

logger = get_logger(__name__)

# Module-level driver cache
_driver = None

# ── User-scoped in-memory graph store ─────────────────────────────────
# Each user has isolated graph data to prevent cross-user leakage
_user_graphs: dict[str, dict] = {}  # user_id -> {nodes: {}, edges: [], papers: {}}


def _get_user_graph(user_id: str) -> dict:
    """Get or create user-scoped graph storage."""
    if not user_id:
        user_id = "__global__"  # Fallback for unauthenticated requests
    if user_id not in _user_graphs:
        _user_graphs[user_id] = {
            "nodes": {},    # name -> {name, entity_type, description, source_paper, category, degree, user_id}
            "edges": [],    # [{source, target, rel_type, description, source_paper, user_id}]
            "papers": {},   # paper_id -> {title, authors, year, user_id}
        }
    return _user_graphs[user_id]


def clear_user_graph(user_id: str) -> None:
    """Clear all graph data for a specific user. Called on new paper upload."""
    if user_id in _user_graphs:
        del _user_graphs[user_id]
        logger.info("Cleared in-memory graph for user: %s", user_id[:8] if user_id else "global")


def _memory_add_node(name: str, entity_type: str, description: str, source_paper: str, user_id: str = "") -> None:
    """Add or update a node in the user-scoped in-memory graph."""
    graph = _get_user_graph(user_id)
    key = normalize_entity_name(name)
    existing = graph["nodes"].get(key)
    if existing is None or len(description) > len(existing.get("description", "")):
        graph["nodes"][key] = {
            "name": name,
            "entity_type": entity_type,
            "description": description,
            "source_paper": source_paper,
            "category": infer_category(Entity(name=name, entity_type=entity_type, description=description)),
            "degree": existing["degree"] if existing else 0,
            "user_id": user_id,
        }


_EDGE_IMPORTANCE_WEIGHTS = {
    "OUTPERFORMS": 1.0,
    "IMPROVES": 0.9,
    "USES": 0.7,
    "EVALUATED_ON": 0.7,
    "EXTENDS": 0.8,
    "BASED_ON": 0.7,
    "COMPARES_WITH": 0.8,
    "CHUNKS_WITH": 0.6,
    "RETRIEVES_WITH": 0.6,
    "EMBEDS_WITH": 0.6,
    "OPTIMIZES": 0.8,
    "LIMITS": 0.5,
    "PART_OF": 0.3,
}


def _compute_edge_weight(rel_type: str, source_paper: str) -> float:
    """Compute semantic edge weight based on relationship type and cross-paper significance."""
    base = _EDGE_IMPORTANCE_WEIGHTS.get(rel_type.upper(), 0.5)
    # Cross-paper edges get a bonus (appear in multiple papers = more significant)
    return base


def _memory_add_edge(source: str, target: str, rel_type: str, description: str, source_paper: str, user_id: str = "") -> None:
    """Add an edge to the user-scoped in-memory graph with semantic weighting."""
    graph = _get_user_graph(user_id)
    weight = _compute_edge_weight(rel_type, source_paper)
    graph["edges"].append({
        "source": source,
        "target": target,
        "rel_type": rel_type,
        "description": description,
        "source_paper": source_paper,
        "weight": weight,
        "user_id": user_id,
    })
    # Increment degree for connected nodes
    src_key = normalize_entity_name(source)
    tgt_key = normalize_entity_name(target)
    if src_key in graph["nodes"]:
        graph["nodes"][src_key]["degree"] = graph["nodes"][src_key].get("degree", 0) + 1
    if tgt_key in graph["nodes"]:
        graph["nodes"][tgt_key]["degree"] = graph["nodes"][tgt_key].get("degree", 0) + 1


# Map of semantic relationship types to valid Neo4j relationship type names
_REL_TYPE_MAP = {
    "uses": "USES",
    "compares_with": "COMPARES_WITH",
    "improves_upon": "IMPROVES_UPON",
    "improves": "IMPROVES",
    "based_on": "BASED_ON",
    "evaluated_on": "EVALUATED_ON",
    "related_to": "RELATED_TO",
    "part_of": "PART_OF",
    "extends": "EXTENDS",
    "implements": "IMPLEMENTS",
    "outperforms": "OUTPERFORMS",
    "applies": "APPLIES",
    "produces": "PRODUCES",
    "requires": "REQUIRES",
    "contradicts": "CONTRADICTS",
    "supports": "SUPPORTS",
    "similar_to": "SIMILAR_TO",
    "chunks_with": "CHUNKS_WITH",
    "retrieves_with": "RETRIEVES_WITH",
    "embeds_with": "EMBEDS_WITH",
    "optimizes": "OPTIMIZES",
    "limits": "LIMITS",
}


def _sanitize_rel_type(raw: str) -> str:
    """Convert a relationship type string to a valid Neo4j relationship type."""
    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")
    mapped = _REL_TYPE_MAP.get(normalized)
    if mapped:
        return mapped
    # Fallback: uppercase and strip non-alphanumeric
    cleaned = re.sub(r"[^A-Z0-9_]", "", normalized.upper())
    return cleaned if cleaned else "RELATED_TO"


def get_driver():
    """Get or create the Neo4j driver."""
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info("Neo4j driver connected to %s", settings.neo4j_uri)
    return _driver


def close_driver():
    """Close the Neo4j driver."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None
        logger.info("Neo4j driver closed")


def init_graph_schema():
    """
    Create indexes and constraints in Neo4j for performance.
    """
    driver = get_driver()
    with driver.session() as session:
        # Create uniqueness constraint on Entity name
        session.run(
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
        # Create index on Paper node
        session.run(
            "CREATE INDEX paper_id_index IF NOT EXISTS "
            "FOR (p:Paper) ON (p.paper_id)"
        )
        logger.info("Graph schema initialised")


def add_entities(entities: list[Entity], user_id: str = "") -> int:
    """
    Add entity nodes to the knowledge graph.
    Merges on name to avoid duplicates.
    Also populates the in-memory graph for fallback visualization.
    
    Args:
        entities: List of entities to add
        user_id: User ID for graph isolation (required for multi-user)
    """
    if not entities:
        return 0

    # Normalize and deduplicate before storage
    entities = deduplicate_entities(entities)

    # Always populate user-scoped in-memory graph
    for entity in entities:
        _memory_add_node(entity.name, entity.entity_type, entity.description, entity.source_paper, user_id)

    # Add category nodes for hierarchy
    category_edges = build_category_edges(entities)
    for cat_name in set(infer_category(e) for e in entities):
        _memory_add_node(cat_name, "category", f"Category: {cat_name}", "", user_id)
    for edge in category_edges:
        _memory_add_edge(edge.source_entity, edge.target_entity, edge.relationship_type, edge.description, edge.source_paper, user_id)

    # Try Neo4j with user_id for isolation
    count = len(entities)
    try:
        driver = get_driver()
        with driver.session() as session:
            for entity in entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $name, user_id: $user_id})
                    SET e.entity_type = $entity_type,
                        e.description = $description,
                        e.source_paper = $source_paper,
                        e.source_chunk_id = $source_chunk_id
                    """,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    source_paper=entity.source_paper,
                    source_chunk_id=entity.source_chunk_id,
                    user_id=user_id,
                )
    except Exception as e:
        logger.warning("Neo4j unavailable for add_entities (using in-memory): %s", e)

    logger.info("Added/merged %d entities to graph for user %s", count, user_id[:8] if user_id else "global")
    return count


def add_relationships(relationships: list[Relationship], user_id: str = "") -> int:
    """
    Add relationship edges between entities using DYNAMIC Neo4j
    relationship types (USES, IMPROVES_UPON, etc.) instead of a
    generic RELATES_TO for everything.
    Also populates the in-memory graph for fallback visualization.
    
    Args:
        relationships: List of relationships to add
        user_id: User ID for graph isolation (required for multi-user)
    """
    if not relationships:
        return 0

    # Normalize and deduplicate
    relationships = deduplicate_relationships(relationships)

    # Always populate user-scoped in-memory graph
    graph = _get_user_graph(user_id)
    for rel in relationships:
        _memory_add_edge(
            rel.source_entity, rel.target_entity,
            rel.relationship_type, rel.description, rel.source_paper, user_id,
        )
        # Ensure source/target nodes exist in memory
        src_key = normalize_entity_name(rel.source_entity)
        tgt_key = normalize_entity_name(rel.target_entity)
        if src_key not in graph["nodes"]:
            _memory_add_node(rel.source_entity, "concept", "", rel.source_paper, user_id)
        if tgt_key not in graph["nodes"]:
            _memory_add_node(rel.target_entity, "concept", "", rel.source_paper, user_id)

    # Try Neo4j with user_id for isolation
    count = len(relationships)
    try:
        driver = get_driver()
        with driver.session() as session:
            for rel in relationships:
                neo4j_type = _sanitize_rel_type(rel.relationship_type)
                session.run(
                    f"""
                    MERGE (a:Entity {{name: $source, user_id: $user_id}})
                    MERGE (b:Entity {{name: $target, user_id: $user_id}})
                    MERGE (a)-[r:{neo4j_type}]->(b)
                    SET r.description = $description,
                        r.source_paper = $source_paper,
                        r.semantic_type = $semantic_type,
                        r.user_id = $user_id
                    """,
                    source=rel.source_entity,
                    target=rel.target_entity,
                    description=rel.description,
                    source_paper=rel.source_paper,
                    semantic_type=rel.relationship_type,
                    user_id=user_id,
                )
    except Exception as e:
        logger.warning("Neo4j unavailable for add_relationships (using in-memory): %s", e)

    logger.info("Added/merged %d relationships to graph for user %s", count, user_id[:8] if user_id else "global")
    return count


def add_paper_node(paper_id: str, title: str, authors: list[str], year: int | None = None, user_id: str = ""):
    """Add a paper reference node and link entities to it."""
    # Always populate user-scoped in-memory store
    graph = _get_user_graph(user_id)
    graph["papers"][paper_id] = {"title": title, "authors": authors, "year": year, "user_id": user_id}
    _memory_add_node(title, "Paper", f"Paper: {title}", paper_id, user_id)

    try:
        driver = get_driver()
        with driver.session() as session:
            session.run(
                """
                MERGE (p:Paper {paper_id: $paper_id})
                SET p.title = $title,
                    p.authors = $authors,
                    p.year = $year
                """,
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
            )

            # Link entities from this paper to the paper node
            session.run(
                """
                MATCH (e:Entity {source_paper: $paper_id})
                MATCH (p:Paper {paper_id: $paper_id})
                MERGE (e)-[:FOUND_IN]->(p)
                """,
                paper_id=paper_id,
            )
    except Exception as e:
        logger.warning("Neo4j unavailable for add_paper_node (using in-memory): %s", e)


def query_related_entities(entity_name: str, max_hops: int = 2) -> list[dict]:
    """
    Multi-hop query: find entities related to a given entity.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH path = (start:Entity {{name: $name}})-[*1..{max_hops}]-(related:Entity)
            RETURN related.name AS name,
                   related.entity_type AS entity_type,
                   related.description AS description,
                   related.source_paper AS source_paper,
                   length(path) AS hops
            ORDER BY hops
            LIMIT 50
            """,
            name=entity_name,
        )
        return [dict(record) for record in result]


def query_cross_paper_connections() -> list[dict]:
    """
    Find entities that appear across multiple papers, revealing cross-paper insights.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)-[:FOUND_IN]->(p:Paper)
            WITH e, collect(DISTINCT p.title) AS papers, count(DISTINCT p) AS paper_count
            WHERE paper_count > 1
            RETURN e.name AS entity,
                   e.entity_type AS entity_type,
                   papers,
                   paper_count
            ORDER BY paper_count DESC
            LIMIT 30
            """
        )
        return [dict(record) for record in result]


def query_entity_graph_summary() -> dict[str, Any]:
    """
    Get a summary of the knowledge graph: node counts, relationship counts, etc.
    """
    driver = get_driver()
    with driver.session() as session:
        entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
        paper_count = session.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

        return {
            "total_entities": entity_count,
            "total_papers": paper_count,
            "total_relationships": rel_count,
        }


def clear_graph():
    """Delete all nodes and relationships in the graph."""
    driver = get_driver()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        logger.info("Knowledge graph cleared")

def clear_user_graph_neo4j(user_id: str) -> None:
    """Clear all Neo4j graph data for a specific user."""
    if not user_id:
        return
    try:
        driver = get_driver()
        with driver.session() as session:
            # Delete all nodes and relationships for this user
            session.run(
                "MATCH (n {user_id: $user_id}) DETACH DELETE n",
                user_id=user_id,
            )
            logger.info("Cleared Neo4j graph for user: %s", user_id[:8])
    except Exception as e:
        logger.warning("Could not clear Neo4j graph for user (non-fatal): %s", e)


def _get_memory_graph_cytoscape(user_id: str = "") -> dict:
    """
    Build Cytoscape JSON from the user-scoped in-memory graph store.
    Used as fallback when Neo4j is unavailable.
    """
    graph = _get_user_graph(user_id)
    nodes = []
    edges = []
    node_id_map: dict[str, str] = {}  # normalized_name -> id

    # Build nodes from user's graph
    for idx, (key, node_data) in enumerate(graph["nodes"].items()):
        node_id = f"mem_{idx}"
        node_id_map[key] = node_id
        name = node_data["name"]
        display_val = name[:37] + "..." if len(name) > 40 else name
        entity_type = node_data.get("entity_type", "concept")
        category = node_data.get("category", "Concepts")
        degree = node_data.get("degree", 0)

        nodes.append({
            "data": {
                "id": node_id,
                "label": display_val,
                "type": "Category" if entity_type == "category" else "Entity",
                "entity_type": entity_type,
                "group": category,
                "full_name": name,
                "description": node_data.get("description", ""),
                "degree": degree,
            }
        })

    # Build edges from user's graph
    for idx, edge_data in enumerate(graph["edges"]):
        src_key = normalize_entity_name(edge_data["source"])
        tgt_key = normalize_entity_name(edge_data["target"])
        src_id = node_id_map.get(src_key)
        tgt_id = node_id_map.get(tgt_key)

        if src_id and tgt_id and src_id != tgt_id:
            rel_type = edge_data.get("rel_type", "RELATED_TO")
            display_label = rel_type.replace("_", " ").lower()
            semantic_weight = edge_data.get("weight", 0.5)
            edges.append({
                "data": {
                    "id": f"edge_{idx}",
                    "source": src_id,
                    "target": tgt_id,
                    "label": display_label,
                    "rel_type": rel_type,
                    "description": edge_data.get("description", ""),
                    "weight": semantic_weight,
                    "source_paper": edge_data.get("source_paper", ""),
                }
            })

    # Filter orphan nodes (keep only those with edges)
    connected_ids = set()
    for e in edges:
        connected_ids.add(e["data"]["source"])
        connected_ids.add(e["data"]["target"])

    connected_nodes = [n for n in nodes if n["data"]["id"] in connected_ids]

    return {"nodes": connected_nodes, "edges": edges}


def get_full_graph_data_cytoscape(user_id: str = "") -> dict:
    """
    Retrieve graph data and export in Cytoscape JSON format.
    
    Args:
        user_id: User ID for filtering — returns only this user's graph data.
                 If empty, returns empty graph (no cross-user leakage).

    IMPROVED:
      - User-scoped graph isolation (multi-user safe)
      - Tries Neo4j first; falls back to in-memory graph
      - Edge labels use semantic_type property (not generic Neo4j type)
      - Edge descriptions included for tooltips
      - Node entity_type included for color-coding
      - Orphan nodes (degree=0) filtered out
      - Node degree included for sizing
      - Category nodes for hierarchical structure
    """
    # If no user_id provided, return empty graph (safe default)
    if not user_id:
        logger.warning("get_full_graph_data_cytoscape called without user_id — returning empty graph")
        return {"nodes": [], "edges": []}
    
    # Try Neo4j first with user filtering
    try:
        driver = get_driver()
        driver.verify_connectivity()

        nodes = {}
        edges = []

        with driver.session() as session:
            # CRITICAL: Filter by user_id to prevent cross-user leakage
            result = session.run(
                """
                MATCH (n {user_id: $user_id})-[r]->(m {user_id: $user_id})
                RETURN n, r, m, type(r) AS rel_type
                LIMIT 1000
                """,
                user_id=user_id,
            )
            connected_node_ids = set()

            for record in result:
                n = record["n"]
                r = record["r"]
                m = record["m"]
                rel_type = record["rel_type"]

                if n is not None:
                    n_id = str(n.element_id)
                    connected_node_ids.add(n_id)
                    if n_id not in nodes:
                        labels = list(n.labels)
                        lbl = labels[0] if labels else "Node"
                        name_field = "title" if lbl == "Paper" else "name"
                        val = n.get(name_field, "Unknown")
                        display_val = val[:37] + "..." if len(val) > 40 else val
                        entity_type = n.get("entity_type", lbl.lower())
                        category = infer_category(Entity(name=val, entity_type=entity_type, description=""))
                        nodes[n_id] = {
                            "data": {
                                "id": n_id,
                                "label": display_val,
                                "type": lbl,
                                "entity_type": entity_type,
                                "group": category,
                                "full_name": val,
                                "description": n.get("description", ""),
                                "degree": 0,
                            }
                        }

                if m is not None:
                    m_id = str(m.element_id)
                    connected_node_ids.add(m_id)
                    if m_id not in nodes:
                        labels = list(m.labels)
                        lbl = labels[0] if labels else "Node"
                        name_field = "title" if lbl == "Paper" else "name"
                        val = m.get(name_field, "Unknown")
                        display_val = val[:37] + "..." if len(val) > 40 else val
                        entity_type = m.get("entity_type", lbl.lower())
                        category = infer_category(Entity(name=val, entity_type=entity_type, description=""))
                        nodes[m_id] = {
                            "data": {
                                "id": m_id,
                                "label": display_val,
                                "type": lbl,
                                "entity_type": entity_type,
                                "group": category,
                                "full_name": val,
                                "description": m.get("description", ""),
                                "degree": 0,
                            }
                        }

                if r is not None and n is not None and m is not None:
                    n_id = str(n.element_id)
                    m_id = str(m.element_id)
                    semantic_label = r.get("semantic_type", rel_type)
                    display_label = semantic_label.replace("_", " ").lower()
                    description = r.get("description", "")

                    edges.append({
                        "data": {
                            "id": str(r.element_id),
                            "source": n_id,
                            "target": m_id,
                            "label": display_label,
                            "rel_type": rel_type,
                            "description": description,
                            "weight": 1,
                        }
                    })

                    if n_id in nodes:
                        nodes[n_id]["data"]["degree"] += 1
                    if m_id in nodes:
                        nodes[m_id]["data"]["degree"] += 1

        connected_nodes = [
            node for nid, node in nodes.items()
            if nid in connected_node_ids
        ]

        if connected_nodes:
            return {"nodes": connected_nodes, "edges": edges}

        # Neo4j connected but empty for this user — try in-memory
        logger.info("Neo4j graph empty for user %s, using in-memory graph", user_id[:8])
        return _get_memory_graph_cytoscape(user_id)

    except Exception as e:
        logger.info("Neo4j unavailable (%s) — using in-memory graph for user %s", e, user_id[:8] if user_id else "unknown")
        return _get_memory_graph_cytoscape(user_id)
