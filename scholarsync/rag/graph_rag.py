"""
GraphRAG engine — stores entities and relationships in Neo4j for
cross-document reasoning, multi-hop queries, and entity linking.
"""

from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import Entity, Relationship

logger = get_logger(__name__)

# Module-level driver cache
_driver = None


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


def add_entities(entities: list[Entity]) -> int:
    """
    Add entity nodes to the knowledge graph.
    Merges on name to avoid duplicates.
    """
    if not entities:
        return 0

    driver = get_driver()
    count = 0

    with driver.session() as session:
        for entity in entities:
            session.run(
                """
                MERGE (e:Entity {name: $name})
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
            )
            count += 1

    logger.info("Added/merged %d entities to graph", count)
    return count


def add_relationships(relationships: list[Relationship]) -> int:
    """
    Add relationship edges between entities.
    Creates entities if they don't exist.
    """
    if not relationships:
        return 0

    driver = get_driver()
    count = 0

    with driver.session() as session:
        for rel in relationships:
            session.run(
                """
                MERGE (a:Entity {name: $source})
                MERGE (b:Entity {name: $target})
                MERGE (a)-[r:RELATES_TO {type: $rel_type}]->(b)
                SET r.description = $description,
                    r.source_paper = $source_paper
                """,
                source=rel.source_entity,
                target=rel.target_entity,
                rel_type=rel.relationship_type,
                description=rel.description,
                source_paper=rel.source_paper,
            )
            count += 1

    logger.info("Added/merged %d relationships to graph", count)
    return count


def add_paper_node(paper_id: str, title: str, authors: list[str], year: int | None = None):
    """Add a paper reference node and link entities to it."""
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

def get_full_graph_data_cytoscape() -> dict:
    """
    Retrieve up to 500 nodes and export them in Cytoscape JSON format.
    Catches connection errors gracefully if Neo4j is offline.
    """
    try:
        driver = get_driver()
        nodes = {}
        edges = []

        with driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN n, r, m LIMIT 500
                """
            )
            for record in result:
                n = record["n"]
                if n is not None:
                    n_id = str(n.element_id)
                    if n_id not in nodes:
                        labels = list(n.labels)
                        lbl = labels[0] if labels else "Node"
                        name_field = "title" if lbl == "Paper" else "name"
                        val = n.get(name_field, "Unknown")
                        if len(val) > 40: val = val[:37] + "..."
                        nodes[n_id] = {
                            "data": {
                                "id": n_id,
                                "label": val,
                                "type": lbl,
                                "group": lbl,
                                "full_name": n.get(name_field, ""),
                                "description": n.get("description", "")
                            }
                        }

                m = record["m"]
                if m is not None:
                    m_id = str(m.element_id)
                    if m_id not in nodes:
                        labels = list(m.labels)
                        lbl = labels[0] if labels else "Node"
                        name_field = "title" if lbl == "Paper" else "name"
                        val = m.get(name_field, "Unknown")
                        if len(val) > 40: val = val[:37] + "..."
                        nodes[m_id] = {
                            "data": {
                                "id": m_id,
                                "label": val,
                                "type": lbl,
                                "group": lbl,
                                "full_name": m.get(name_field, ""),
                                "description": m.get("description", "")
                            }
                        }

                r = record["r"]
                if r is not None and n is not None and m is not None:
                    edges.append({
                        "data": {
                            "id": str(r.element_id),
                            "source": str(n.element_id),
                            "target": str(m.element_id),
                            "label": r.type
                        }
                    })

        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }
    except Exception as e:
        logger.warning(f"Neo4j offline or unreachable: returning empty graph. Error: {e}")
        return {"nodes": [], "edges": []}

