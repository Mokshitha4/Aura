# agent.py
# This file contains the AuraAgent class, which now acts as a collection of "tools".
# The orchestrator will decide which of these tools to use.

import logging
import json
import requests
import os
import wikipediaapi
from aura_brain import AuraBrain

# --- CONFIGURATION ---
QLOO_API_KEY = os.getenv("QLOO_API_KEY")
QLOO_SEARCH_URL = 'https://hackathon.api.qloo.com/v2/search'
QLOO_INSIGHTS_URL = 'https://hackathon.api.qloo.com/v2/insights'

class AuraAgent:
    """
    The agent that holds the specialist "tools" for interacting with the world and the knowledge base.
    """
    def __init__(self, brain: AuraBrain):
        if not isinstance(brain, AuraBrain):
            raise TypeError("Agent must be initialized with an instance of AuraBrain.")
        self.brain = brain
        self.openai_client = brain.openai_client
        self.wiki_api = wikipediaapi.Wikipedia('AuraPersonalAgent/1.0', 'mokshithamandadi14@gmail.com')
        logging.info("AuraAgent initialized with tools.")

    # --- TOOL 1: Update Knowledge Base ---
    # *** EDIT APPLIED: This tool is now simpler. It no longer calls an LLM. ***
    # It expects to receive a pre-extracted graph from the supervisor.
    def update_knowledge_base(self, extracted_graph: dict):
        """
        Tool to save a pre-extracted graph of entities and relationships
        to the personal knowledge base.
        The 'extracted_graph' should be a dictionary with 'nodes' and 'edges' keys.
        Returns a confirmation message.
        """
        logging.info(f"TOOL CALLED: update_knowledge_base with pre-extracted graph.")
        
        nodes_data = extracted_graph.get("nodes", [])
        edges_data = extracted_graph.get("edges", [])

        if not nodes_data:
            return "No valid information was extracted to save."

        node_id_map = {}
        for node in nodes_data:
            node_id = self.brain.find_or_create_node(node['content'], node['type'])
            node_id_map[node['content']] = node_id

        for edge in edges_data:
            source_id = node_id_map.get(edge['source'])
            target_id = node_id_map.get(edge['target'])
            if source_id is not None and target_id is not None:
                self.brain.add_edge(source_id, target_id, edge['relationship'])

        self.brain.save()
        return f"Successfully updated the knowledge base with information about: {', '.join([n['content'] for n in nodes_data])}."

    # --- TOOL 2: Query Knowledge Base ---
    def query_knowledge_base(self, query: str):
        """
        Tool to perform a hybrid search on the personal knowledge base.
        Returns a string summary of the findings.
        """
        logging.info(f"TOOL CALLED: query_knowledge_base with query: '{query}'")
        search_context = self.brain.hybrid_search(query, k=3)
        if not search_context:
            return "No relevant information found in the personal knowledge base."

        context_str = "\n".join([f"- Found '{data['content']}' (Type: {data['type']}). It is factually connected to: {[neighbor['content'] for neighbor in data['neighborhood']]}" for node_id, data in search_context.items()])
        return f"Found the following relevant information in the knowledge base:\n{context_str}"

    # --- TOOL 3: External Search ---
    def external_search(self, query: str):
        """
        Tool to search for general information on the web using Wikipedia.
        Returns a summary of the topic.
        """
        logging.info(f"TOOL CALLED: external_search with query: '{query}'")
        try:
            page = self.wiki_api.page(query)
            if page.exists():
                return f"Wikipedia summary for '{query}':\n{page.summary[0:500]}..." # Return first 500 chars
            else:
                return f"Could not find a Wikipedia page for '{query}'."
        except Exception as e:
            logging.error(f"Wikipedia search failed: {e}")
            return "Sorry, I had trouble searching online right now."

    # --- TOOL 4: Qloo Enrichment ---
    def qloo_enrichment(self, entity_name: str, entity_type: str):
        """
        Tool to find culturally related recommendations for a given entity (like a movie, book, or artist)
        and add them to the knowledge base.
        """
        logging.info(f"TOOL CALLED: qloo_enrichment for entity: '{entity_name}'")
        
        source_node_id, _ = self.brain.get_node_by_content(entity_name)
        if source_node_id is None:
            return f"Error: Could not find the entity '{entity_name}' in the knowledge base to enrich."

        recommendations = self._get_qloo_recommendations(entity_name, entity_type)
        if not recommendations:
            return f"Found no cultural recommendations for '{entity_name}'."
            
        added_recs = []
        for rec in recommendations:
            rec_name = rec.get("name")
            rec_category = rec.get("category", "Recommendation").capitalize()
            if rec_name:
                target_node_id = self.brain.find_or_create_node(rec_name, rec_category)
                if target_node_id is not None:
                    self.brain.add_edge(source_node_id, target_node_id, f"QLOO_RECOMMENDS_{rec_category.upper()}")
                    added_recs.append(rec_name)
        
        self.brain.save()
        return f"Successfully added cultural recommendations for '{entity_name}': {', '.join(added_recs)}."


    # --- Helper methods ---
    # *** EDIT APPLIED: The _extract_graph_from_text method has been removed. ***
    # This logic now belongs in the supervisor/orchestrator.

    def _get_qloo_entity_id(self, query: str, entity_type: str):
        """Step 1 of Qloo API: Search for an entity's unique ID."""
        if not QLOO_API_KEY: return None
        params = {'query': query, 'types': f'urn:entity:{entity_type}'}
        headers = {"accept": "application/json", "X-Api-Key": QLOO_API_KEY}
        try:
            response = requests.get(QLOO_SEARCH_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("results"):
                entity_id = data["results"][0].get("entity_id")
                logging.info(f"Qloo found Entity ID '{entity_id}' for query '{query}'.")
                return entity_id
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Qloo API search error: {e}")
            return None

    def _get_qloo_recommendations(self, entity_name: str, entity_type: str):
        """Calls the Qloo API in a two-step process to get recommendations."""
        qloo_entity_id = self._get_qloo_entity_id(entity_name, entity_type)
        if not qloo_entity_id: return []

        all_recommendations = []
        output_types = ["place", "book", "music"]
        for output_type in output_types:
            payload = {"filter.type": f"urn:entity:{output_type}", "signal.interests.entities": qloo_entity_id, "limit": 2}
            headers = {"accept": "application/json", "X-Api-Key": QLOO_API_KEY}
            try:
                response = requests.get(QLOO_INSIGHTS_URL, headers=headers, params=payload)
                response.raise_for_status()
                recommendations = response.json().get("results", [])
                for rec in recommendations:
                    rec['category'] = output_type # Add category for context
                all_recommendations.extend(recommendations)
            except requests.exceptions.RequestException as e:
                logging.warning(f"Could not fetch Qloo recommendations for '{entity_name}' (type: {output_type}): {e}")
        return all_recommendations
