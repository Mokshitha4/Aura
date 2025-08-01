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
    def update_knowledge_base(self, text: str):
        """
        Tool to process unstructured text, extract entities and relationships,
        enrich them with Qloo, and save them to the knowledge base.
        Returns a confirmation message.
        """
        logging.info(f"TOOL CALLED: update_knowledge_base with text: '{text[:50]}...'")
        extracted_graph = self._extract_graph_from_text(text)
        if not extracted_graph or not extracted_graph.get("nodes"):
            self.brain.add_node(text, "Note")
            self.brain.save()
            return "Successfully saved the information as a single note."

        nodes_data = extracted_graph.get("nodes", [])
        edges_data = extracted_graph.get("edges", [])

        node_id_map = {}
        for node in nodes_data:
            node_id = self.brain.find_or_create_node(node['content'], node['type'])
            node_id_map[node['content']] = node_id

        for edge in edges_data:
            source_id = node_id_map.get(edge['source'])
            target_id = node_id_map.get(edge['target'])
            if source_id is not None and target_id is not None:
                self.brain.add_edge(source_id, target_id, edge['relationship'])

        if nodes_data:
            primary_entity_content = nodes_data[0]['content']
            primary_entity_type = nodes_data[0].get('type', 'person').lower()
            primary_entity_id = node_id_map.get(primary_entity_content)
            if primary_entity_id is not None:
                self._enrich_node_with_qloo(primary_entity_id, primary_entity_content, primary_entity_type)

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

    # --- Helper methods ---
    def _extract_graph_from_text(self, text: str):
        """Uses an LLM to perform the core information extraction task."""
        prompt = f"""
        You are an information extraction system. Analyze the following text and identify the key entities (nodes) and the factual relationships (edges) between them.
        Return your answer as a single, valid JSON object with two keys: "nodes" and "edges".
        
        - Each node must have a "content" (string) and a "type" (string, e.g., Person, Location, Project, Movie, Note, Book, Artist).
        - Each edge must have a "source" (string, matching a node's content), a "target" (string, matching a node's content), and a "relationship" (string, in uppercase, e.g., LIVES_IN, RECOMMENDED, MENTIONS).
        - If no clear entities or relationships are found, return an empty JSON object.

        Text: "{text}"
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Failed during LLM graph extraction: {e}")
            return None

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

    def _enrich_node_with_qloo(self, node_id: int, content: str, entity_type: str):
        """Calls the Qloo API in a two-step process and adds recommendations."""
        qloo_entity_id = self._get_qloo_entity_id(content, entity_type)
        if not qloo_entity_id: return

        output_types = ["place", "book", "music"]
        for output_type in output_types:
            payload = {"filter.type": f"urn:entity:{output_type}", "signal.interests.entities": qloo_entity_id, "limit": 2}
            headers = {"accept": "application/json", "X-Api-Key": QLOO_API_KEY}
            try:
                response = requests.get(QLOO_INSIGHTS_URL, headers=headers, params=payload)
                response.raise_for_status()
                recommendations = response.json().get("results", [])
                for rec in recommendations:
                    rec_name = rec.get("name")
                    rec_category = output_type.capitalize()
                    if rec_name:
                        target_node_id = self.brain.find_or_create_node(rec_name, rec_category)
                        if target_node_id is not None:
                            self.brain.add_edge(node_id, target_node_id, f"RECOMMENDS_{rec_category.upper()}")
            except requests.exceptions.RequestException as e:
                logging.warning(f"Could not fetch Qloo recommendations for '{content}' (type: {output_type}): {e}")
