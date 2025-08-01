# aura_brain.py
# This file contains the AuraBrain class, the core of the personal knowledge base.
# It uses NetworkX for the factual graph and FAISS (with HNSW) for semantic search.


import os
import networkx as nx
import faiss
import numpy as np
import openai
import logging
from datetime import datetime

# --- CONFIGURATION ---
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # Must match the model's output dimension

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AuraBrain:
    """
    Manages a dynamic knowledge base using a hybrid approach:
    - NetworkX: Stores the explicit, factual graph of nodes and relationships.
    - FAISS (HNSW): Provides a high-speed, dynamic index for semantic search.
    """
    def __init__(self, graph_path="aura_graph.gml", index_path="aura_index.faiss"):
        self.graph_path = graph_path
        self.index_path = index_path
        self.next_node_id = 0

        # Initialize OpenAI client
        try:
            self.openai_client = openai.OpenAI()
            logging.info("OpenAI client initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client. Is OPENAI_API_KEY set? Error: {e}")
            raise

        # Load or initialize the factual graph (NetworkX)
        if os.path.exists(self.graph_path):
            self.graph = nx.read_gml(self.graph_path)
            # Ensure node IDs are integers for FAISS mapping
            self.graph = nx.convert_node_labels_to_integers(self.graph)
            self.next_node_id = max(self.graph.nodes) + 1 if self.graph.nodes else 0
            logging.info(f"Loaded existing graph from {self.graph_path}. Next node ID: {self.next_node_id}")
        else:
            self.graph = nx.DiGraph() # Directed Graph
            logging.info("Created a new, empty graph.")

        # Load or initialize the semantic search index (FAISS)
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            logging.info(f"Loaded existing FAISS index from {self.index_path}.")
        else:
            # Using HNSW for a fast, dynamic index. IndexIDMap lets us use our own node IDs.
            self.index = faiss.IndexIDMap(faiss.IndexHNSWFlat(EMBEDDING_DIM, 32))
            logging.info("Created a new, empty FAISS index with HNSW.")
            # If the graph exists but the index doesn't, we should rebuild it.
            if len(self.graph) > 0:
                self._rebuild_index_from_graph()

    def get_embedding(self, text: str):
        """Generates a vector embedding for a given piece of text."""
        try:
            text = text.replace("\n", " ")
            response = self.openai_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error getting embedding from OpenAI: {e}")
            return None

    def add_node(self, content: str, node_type: str, **kwargs):
        """
        Adds a new node to both the factual graph and the semantic index.
        Returns the ID of the newly created node.
        """
        node_id = self.next_node_id
        embedding = self.get_embedding(content)
        if embedding is None:
            return None

        # 1. Add to NetworkX graph with all metadata
        self.graph.add_node(node_id, content=content, type=node_type, timestamp=datetime.now().isoformat(), **kwargs)

        # 2. Add to FAISS index for semantic search
        embedding_np = np.array([embedding]).astype('float32')
        ids_np = np.array([node_id])
        self.index.add_with_ids(embedding_np, ids_np)

        logging.info(f"Added node {node_id} ('{content[:30]}...') of type '{node_type}'")
        self.next_node_id += 1
        return node_id

    def get_node_by_content(self, content: str):
        """Finds a node by its exact content string."""
        for node_id, data in self.graph.nodes(data=True):
            if data.get('content') == content:
                return node_id, data
        return None, None

    def find_or_create_node(self, content: str, node_type: str, **kwargs):
        """
        A more intelligent way to add nodes. It first searches for an existing node
        with the same content. If found, it returns that node's ID. If not, it creates a new one.
        """
        node_id, _ = self.get_node_by_content(content)
        if node_id is not None:
            logging.info(f"Found existing node {node_id} for content '{content}'.")
            return node_id
        else:
            return self.add_node(content, node_type, **kwargs)

    def add_edge(self, source_id: int, target_id: int, relationship: str):
        """Adds a factual, directed relationship between two nodes in the graph."""
        if self.graph.has_node(source_id) and self.graph.has_node(target_id):
            self.graph.add_edge(source_id, target_id, label=relationship)
            logging.info(f"Added edge: {source_id} -[{relationship}]-> {target_id}")
        else:
            logging.warning(f"Could not create edge. Node ID not found. Source: {source_id}, Target: {target_id}")

    def hybrid_search(self, query_text: str, k: int = 3):
        """
        Performs a powerful hybrid search:
        1. Finds the most semantically similar nodes using FAISS.
        2. Retrieves their factual neighborhood from NetworkX.
        """
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            return {}

        query_embedding_np = np.array([query_embedding]).astype('float32')

        # 1. Fast Semantic Search (FAISS)
        # This returns the distances and the IDs of the most similar vectors
        distances, ids = self.index.search(query_embedding_np, k)

        if ids.size == 0:
            return {}

        # 2. Deep Contextual Traversal (NetworkX)
        results = {}
        for node_id in ids[0]:
            if self.graph.has_node(node_id):
                node_data = self.graph.nodes[node_id]
                neighborhood = []
                # Find all connected nodes (both incoming and outgoing)
                for neighbor_id in nx.all_neighbors(self.graph, node_id):
                    edge_data = self.graph.get_edge_data(node_id, neighbor_id) or self.graph.get_edge_data(neighbor_id, node_id)
                    neighbor_data = self.graph.nodes[neighbor_id]
                    neighborhood.append({
                        "id": neighbor_id,
                        "content": neighbor_data.get('content'),
                        "type": neighbor_data.get('type'),
                        "relationship": edge_data.get('label')
                    })
                results[node_id] = {
                    "content": node_data.get('content'),
                    "type": node_data.get('type'),
                    "neighborhood": neighborhood
                }
        return results

    def save(self):
        """Saves the graph and the index to their respective files."""
        try:
            nx.write_gml(self.graph, self.graph_path)
            logging.info(f"Graph saved to {self.graph_path}")
            faiss.write_index(self.index, self.index_path)
            logging.info(f"Index saved to {self.index_path}")
        except Exception as e:
            logging.error(f"Error saving files: {e}")
            
    def _rebuild_index_from_graph(self):
        """(Helper) Rebuilds the FAISS index if it's out of sync with the graph."""
        logging.info("Attempting to rebuild FAISS index from existing graph...")
        node_ids = []
        embeddings = []
        for node_id, data in self.graph.nodes(data=True):
            if 'content' in data:
                embedding = self.get_embedding(data['content'])
                if embedding:
                    node_ids.append(node_id)
                    embeddings.append(embedding)
        
        if embeddings:
            embeddings_np = np.array(embeddings).astype('float32')
            ids_np = np.array(node_ids)
            self.index.add_with_ids(embeddings_np, ids_np)
            logging.info(f"Successfully rebuilt index with {len(node_ids)} nodes.")


if __name__ == '__main__':
    # --- DEMONSTRATION ---
    print("--- Initializing Aura's Brain ---")
    # This will create the files if they don't exist, or load them if they do.
    brain = AuraBrain()

    # Clear the brain for a fresh demo (optional)
    if len(brain.graph) > 0:
        print("\n--- Clearing existing brain for fresh demo ---")
        brain = AuraBrain(graph_path="aura_graph_new.gml", index_path="aura_index_new.faiss")


    print("\n--- Populating the Knowledge Base using 'find_or_create_node' ---")
    # Simulate user activity and building the graph intelligently.
    
    # Add a new note and the entities it mentions
    note1_id = brain.add_node("Just finished watching the movie Dune, the visuals were incredible.", "Note")
    movie1_id = brain.find_or_create_node("Dune (2021 film)", "Movie")
    person1_id = brain.find_or_create_node("Denis Villeneuve", "Person")
    
    if all([note1_id is not None, movie1_id is not None, person1_id is not None]):
        brain.add_edge(note1_id, movie1_id, "MENTIONS")
        brain.add_edge(person1_id, movie1_id, "DIRECTED")

    # Add another note about a trip
    note2_id = brain.add_node("Planning a trip to Tokyo next fall. Need to book flights soon.", "Note")
    location1_id = brain.find_or_create_node("Tokyo", "Location")
    
    if all([note2_id is not None, location1_id is not None]):
        brain.add_edge(note2_id, location1_id, "MENTIONS")

    # Add a third note that mentions an existing entity ('Tokyo')
    note3_id = brain.add_node("My friend Sarah, who lives in Tokyo, recommended a great book about sci-fi world-building.", "Note")
    person2_id = brain.find_or_create_node("Sarah", "Person")
    # This time, find_or_create_node will FIND the existing 'Tokyo' node instead of creating a new one.
    existing_location_id = brain.find_or_create_node("Tokyo", "Location")

    if all([note3_id is not None, person2_id is not None, existing_location_id is not None]):
        brain.add_edge(note3_id, person2_id, "MENTIONS")
        brain.add_edge(person2_id, existing_location_id, "LIVES_IN")


    print("\n--- Performing a Hybrid Search ---")
    # The user asks a question. The system finds semantically similar nodes (FAISS)
    # and then explores their factual neighborhood (NetworkX).
    
    query = "What are some of my recent creative inspirations?"
    print(f"QUERY: '{query}'")
    
    search_results = brain.hybrid_search(query)

    print("\n--- SEARCH RESULTS ---")
    if not search_results:
        print("No relevant results found.")
    else:
        for node_id, data in search_results.items():
            print(f"\n[+] Found semantically similar node '{data['content']}' (Type: {data['type']})")
            if data['neighborhood']:
                print("    - It is factually connected to:")
                for neighbor in data['neighborhood']:
                    print(f"      - '{neighbor['content']}' (Type: {neighbor['type']}) via relationship '{neighbor['relationship']}'")
            else:
                print("    - It has no direct factual connections yet.")

    print("\n--- Saving Brain to Disk ---")
    brain.save()
