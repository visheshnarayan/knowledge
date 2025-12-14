import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import networkx as nx
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoModelForSeq2SeqLM
import re


class GraphBuilder:
    def __init__(self, data_config, graph_config):
        self.data_config = data_config
        self.graph_config = graph_config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.graph_config["chunking"]["size"],
            chunk_overlap=self.graph_config["chunking"]["overlap"],
        )
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            self.graph_config["embedding"]["model"]
        )
        self.embedding_model = AutoModel.from_pretrained(
            self.graph_config["embedding"]["model"]
        )
        self.similarity_metric = self.graph_config["similarity"]["metric"]
        self.similarity_threshold = self.graph_config["similarity"]["threshold"]

        # Load REBEL model and tokenizer directly
        self.triplet_tokenizer = AutoTokenizer.from_pretrained(
            self.graph_config["triplet_extraction"]["model"]
        )
        self.triplet_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.graph_config["triplet_extraction"]["model"]
        )
        self.cosine_graph = nx.Graph()
        self.triplets_graph = nx.Graph()

    def build(self):
        print("Building graphs...")
        self._chunk_documents()
        self._embed_nodes()
        self._connect_nodes()
        self._extract_triplets()
        print(
            f"Cosine graph built with {self.cosine_graph.number_of_nodes()} nodes and {self.cosine_graph.number_of_edges()} edges."
        )
        print(
            f"Triplets graph built with {self.triplets_graph.number_of_nodes()} nodes and {self.triplets_graph.number_of_edges()} edges."
        )
        return self.cosine_graph, self.triplets_graph

    def _chunk_documents(self):
        print("Chunking documents...")
        input_dir = self.data_config["input_dir"]
        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(input_dir, filename)
                with open(filepath, "r") as f:
                    text = f.read()
                chunks = self.text_splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    node_id = f"{filename}_chunk_{i}"
                    self.cosine_graph.add_node(
                        node_id, content=chunk, type="chunk", filename=filename
                    )
                    self.triplets_graph.add_node(
                        node_id, content=chunk, type="chunk", filename=filename
                    )

    def _embed_nodes(self):
        print("Embedding nodes...")
        embeddings_cache = {}
        for node_id, node_data in self.cosine_graph.nodes(data=True):
            if node_data["type"] == "chunk":
                # Generate embeddings using transformers AutoModel
                inputs = self.embedding_tokenizer(
                    node_data["content"],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                )
                with torch.no_grad():
                    embedding = (
                        self.embedding_model(**inputs)
                        .last_hidden_state.mean(dim=1)
                        .squeeze()
                        .tolist()
                    )
                node_data["embedding"] = embedding
                embeddings_cache[node_id] = embedding

        # Also add embeddings to the triplets graph
        for node_id, node_data in self.triplets_graph.nodes(data=True):
            if node_data["type"] == "chunk":
                if node_id in embeddings_cache:
                    node_data["embedding"] = embeddings_cache[node_id]
                else:
                    # This case should ideally not be hit if nodes are the same
                    inputs = self.embedding_tokenizer(
                        node_data["content"],
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                    )
                    with torch.no_grad():
                        embedding = (
                            self.embedding_model(**inputs)
                            .last_hidden_state.mean(dim=1)
                            .squeeze()
                            .tolist()
                        )
                    node_data["embedding"] = embedding

    def _connect_nodes(self):
        print("Connecting nodes...")
        if self.similarity_metric != "cosine":
            raise ValueError(
                f"Similarity metric '{self.similarity_metric}' is not supported. Please use 'cosine'."
            )

        chunk_nodes = [
            n for n in self.cosine_graph.nodes(data=True) if n[1]["type"] == "chunk"
        ]
        for i in range(len(chunk_nodes)):
            for j in range(i + 1, len(chunk_nodes)):
                node1_id, node1_data = chunk_nodes[i]
                node2_id, node2_data = chunk_nodes[j]
                embedding1 = np.array(node1_data["embedding"])
                embedding2 = np.array(node2_data["embedding"])
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                if similarity > self.similarity_threshold:
                    self.cosine_graph.add_edge(
                        node1_id,
                        node2_id,
                        weight=similarity,
                        type="semantic_similarity",
                    )

    def _extract_triplets(self):
        print("Extracting triplets...")
        # Collect chunk nodes first to avoid RuntimeError during iteration
        chunk_nodes_to_process = [
            (node_id, node_data)
            for node_id, node_data in self.triplets_graph.nodes(data=True)
            if node_data["type"] == "chunk"
        ]

        for node_id, node_data in chunk_nodes_to_process:
            text = node_data["content"]
            try:
                # Prepare input for REBEL model
                model_inputs = self.triplet_tokenizer(
                    text, max_length=256, truncation=True, return_tensors="pt"
                )
                generated_tokens = self.triplet_model.generate(
                    model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    max_length=100,
                    num_beams=5,
                    do_sample=False,
                )
                decoded_triplets = self.triplet_tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=False
                )[0]
                # Remove <s> and </s> tokens
                cleaned_triplets = (
                    decoded_triplets.replace("<s>", "").replace("</s>", "").strip()
                )

                # Parse the linearized triplets using a single regex
                parsed_triplets = []
                # Regex to find all triplets: <triplet> subject <subj> predicate <obj> object </triplet>
                triplet_pattern = r"<triplet>(.*?)\s*<subj>(.*?)\s*<obj>(.*?)</triplet>"

                # Parse the linearized triplets using a single regex
                parsed_triplets = []
                # Regex to find all triplets: <triplet> subject <subj> predicate <obj> object
                triplet_pattern = r"<triplet>(.*?)\s*<subj>(.*?)\s*<obj>(.*?)"

                for match in re.finditer(triplet_pattern, cleaned_triplets):
                    subject = match.group(1).strip()
                    predicate = match.group(2).strip()
                    obj = match.group(3).strip()
                    parsed_triplets.append((subject, predicate, obj))

                for i, triplet in enumerate(parsed_triplets):
                    subject, predicate, obj = triplet
                    triplet_node_id = f"{node_id}_triplet_{i}"
                    self.triplets_graph.add_node(
                        triplet_node_id,
                        content=f"({subject}, {predicate}, {obj})",
                        type="triplet",
                    )
                    self.triplets_graph.add_edge(
                        node_id,
                        triplet_node_id,
                        type="contains_triplet",
                        triplet_info=f"({subject}, {predicate}, {obj})",
                    )
            except Exception as e:
                print(f"Error extracting triplets from chunk {node_id}: {e}")
