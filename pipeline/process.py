#!/usr/bin/env python3
"""
Hidden Connections - Project 4 (Maximum Features)
ML Pipeline with CLIP/LLM Embedding Support

Supports multiple embedding backends:
- sentence-transformers (fast, lightweight)
- BGE (BAAI/bge-* models - excellent quality)
- Nomic (nomic-embed-text)
- CLIP (OpenAI CLIP - multimodal)
- OpenAI API (text-embedding-3-*)
- Instructor (custom instruction prefix)
"""

import argparse
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
import umap

# Rich console for pretty output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None


# =============================================================================
# EMBEDDING BACKENDS
# =============================================================================

class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class SentenceTransformerBackend(EmbeddingBackend):
    """Sentence Transformers backend (all-MiniLM, all-mpnet, etc.)"""

    def __init__(self, model_name: str, device: str = "auto", normalize: bool = True):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=self._resolve_device(device))
        self.normalize = normalize
        self._dimension = self.model.get_sentence_embedding_dimension()
        logging.info(f"Loaded SentenceTransformer: {model_name} (dim={self._dimension})")

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize
        )
        return np.array(embeddings)

    @property
    def dimension(self) -> int:
        return self._dimension


class BGEBackend(EmbeddingBackend):
    """BAAI/BGE models backend - high quality embeddings."""

    def __init__(self, model_name: str, device: str = "auto", normalize: bool = True,
                 instruction: str = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=self._resolve_device(device))
        self.normalize = normalize
        self.instruction = instruction
        self._dimension = self.model.get_sentence_embedding_dimension()
        logging.info(f"Loaded BGE model: {model_name} (dim={self._dimension})")

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        # BGE models work best with instruction prefix for queries
        if self.instruction:
            texts = [f"{self.instruction} {t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize
        )
        return np.array(embeddings)

    @property
    def dimension(self) -> int:
        return self._dimension


class NomicBackend(EmbeddingBackend):
    """Nomic AI embedding models."""

    def __init__(self, model_name: str, device: str = "auto", normalize: bool = True):
        from sentence_transformers import SentenceTransformer
        # Nomic models require trust_remote_code
        self.model = SentenceTransformer(
            model_name,
            device=self._resolve_device(device),
            trust_remote_code=True
        )
        self.normalize = normalize
        self._dimension = self.model.get_sentence_embedding_dimension()
        logging.info(f"Loaded Nomic model: {model_name} (dim={self._dimension})")

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        # Nomic models use task prefixes
        prefixed_texts = [f"search_document: {t}" for t in texts]
        embeddings = self.model.encode(
            prefixed_texts,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize
        )
        return np.array(embeddings)

    @property
    def dimension(self) -> int:
        return self._dimension


class CLIPBackend(EmbeddingBackend):
    """OpenAI CLIP models via open-clip."""

    def __init__(self, model_name: str, device: str = "auto", normalize: bool = True):
        import open_clip
        import torch

        self.device = self._resolve_device(device)
        self.normalize = normalize

        # Parse model name (e.g., "openai/clip-vit-large-patch14" -> "ViT-L-14", "openai")
        model_arch, pretrained = self._parse_model_name(model_name)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_arch, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(model_arch)
        self._dimension = self.model.text_projection.shape[1] if hasattr(self.model, 'text_projection') else 512

        logging.info(f"Loaded CLIP model: {model_name} (dim={self._dimension})")

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _parse_model_name(self, model_name: str) -> tuple[str, str]:
        """Parse model name to open-clip format."""
        model_map = {
            "openai/clip-vit-base-patch32": ("ViT-B-32", "openai"),
            "openai/clip-vit-base-patch16": ("ViT-B-16", "openai"),
            "openai/clip-vit-large-patch14": ("ViT-L-14", "openai"),
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": ("ViT-H-14", "laion2b_s32b_b79k"),
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": ("ViT-bigG-14", "laion2b_s39b_b160k"),
        }
        if model_name in model_map:
            return model_map[model_name]
        # Default: assume it's already in open-clip format
        parts = model_name.split("/")
        if len(parts) == 2:
            return parts[1], parts[0]
        return model_name, "openai"

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        import torch
        from tqdm import tqdm

        all_embeddings = []
        batch_size = 32

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding with CLIP")

        with torch.no_grad():
            for i in iterator:
                batch = texts[i:i + batch_size]
                tokens = self.tokenizer(batch).to(self.device)
                embeddings = self.model.encode_text(tokens)

                if self.normalize:
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIBackend(EmbeddingBackend):
    """OpenAI API embeddings (text-embedding-3-small/large)."""

    def __init__(self, model_name: str, normalize: bool = True, **kwargs):
        from openai import OpenAI

        self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        self.model_name = model_name
        self.normalize = normalize

        # Dimension mapping
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimension = dim_map.get(model_name, 1536)
        logging.info(f"Initialized OpenAI embeddings: {model_name} (dim={self._dimension})")

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        from tqdm import tqdm

        all_embeddings = []
        batch_size = 100  # OpenAI allows up to 2048

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding with OpenAI")

        for i in iterator:
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            batch_embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embeddings)

        embeddings = np.array(all_embeddings)

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


class InstructorBackend(EmbeddingBackend):
    """Instructor embeddings with custom instructions."""

    def __init__(self, model_name: str, device: str = "auto", normalize: bool = True,
                 instruction: str = "Represent the text:"):
        from InstructorEmbedding import INSTRUCTOR

        self.device = self._resolve_device(device)
        self.model = INSTRUCTOR(model_name, device=self.device)
        self.normalize = normalize
        self.instruction = instruction
        self._dimension = 768  # Instructor models are typically 768-dim
        logging.info(f"Loaded Instructor model: {model_name} (dim={self._dimension})")

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        # Instructor format: [[instruction, text], ...]
        inputs = [[self.instruction, t] for t in texts]
        embeddings = self.model.encode(inputs, show_progress_bar=show_progress)

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return np.array(embeddings)

    @property
    def dimension(self) -> int:
        return self._dimension


def create_embedding_backend(config: dict) -> EmbeddingBackend:
    """Factory function to create the appropriate embedding backend."""
    emb_config = config.get("embedding", {})
    emb_type = emb_config.get("type", "sentence-transformers")
    model_name = emb_config.get("model", "all-MiniLM-L6-v2")
    device = emb_config.get("device", "auto")
    normalize = emb_config.get("normalize", True)
    instruction = emb_config.get("instruction", None)

    backends = {
        "sentence-transformers": lambda: SentenceTransformerBackend(model_name, device, normalize),
        "bge": lambda: BGEBackend(model_name, device, normalize, instruction),
        "nomic": lambda: NomicBackend(model_name, device, normalize),
        "clip": lambda: CLIPBackend(model_name, device, normalize),
        "openai": lambda: OpenAIBackend(model_name, normalize),
        "instructor": lambda: InstructorBackend(model_name, device, normalize, instruction or "Represent the text:"),
    }

    if emb_type not in backends:
        raise ValueError(f"Unknown embedding type: {emb_type}. Supported: {list(backends.keys())}")

    return backends[emb_type]()


# =============================================================================
# CLUSTERING BACKENDS
# =============================================================================

def create_clusterer(config: dict):
    """Create clustering model based on config."""
    cluster_config = config.get("clustering", {})
    algorithm = cluster_config.get("algorithm", "kmeans")
    n_clusters = cluster_config.get("n_clusters", 5)
    random_state = cluster_config.get("random_state", 42)

    if algorithm == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    elif algorithm == "agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == "hdbscan":
        import hdbscan
        return hdbscan.HDBSCAN(
            min_cluster_size=cluster_config.get("min_cluster_size", 5),
            min_samples=cluster_config.get("min_samples", 3)
        )
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")


# =============================================================================
# DATA PROCESSING
# =============================================================================

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """Setup logging based on config."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )


def load_data(csv_path: Path, text_fields: list, categorical_fields: list,
              max_chars: int) -> pd.DataFrame:
    """Load and validate CSV data."""
    df = pd.read_csv(csv_path)

    required_cols = ['id'] + text_fields + categorical_fields + ['nickname']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Fill missing nicknames
    df['nickname'] = df['nickname'].fillna('')

    # Truncate text fields
    for field in text_fields:
        df[field] = df[field].astype(str).str[:max_chars]

    logging.info(f"Loaded {len(df)} participants from {csv_path}")
    return df


def create_combined_text(df: pd.DataFrame, text_fields: list,
                         field_labels: dict) -> list[str]:
    """Concatenate text fields into single strings with labels."""
    combined = []
    for _, row in df.iterrows():
        parts = []
        for i, field in enumerate(text_fields, 1):
            label = field_labels.get(field, field)
            parts.append(f"Q{i} ({label}): {row[field]}")
        combined.append("\n".join(parts))
    return combined


def reduce_dimensions(embeddings: np.ndarray, config: dict) -> np.ndarray:
    """Reduce embeddings to 2D using UMAP."""
    umap_config = config.get("umap", {})

    logging.info("Applying UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_neighbors=umap_config.get("n_neighbors", 15),
        min_dist=umap_config.get("min_dist", 0.1),
        metric=umap_config.get("metric", "cosine"),
        n_components=umap_config.get("n_components", 2),
        random_state=umap_config.get("random_state", 42)
    )

    coords_2d = reducer.fit_transform(embeddings)
    logging.info(f"Reduced to shape: {coords_2d.shape}")
    return coords_2d


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """Normalize coordinates to [-1, 1] range using sklearn."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(coords)


def cluster_points(embeddings: np.ndarray, config: dict) -> np.ndarray:
    """Cluster embeddings."""
    clusterer = create_clusterer(config)
    logging.info(f"Clustering with {config['clustering']['algorithm']}...")

    labels = clusterer.fit_predict(embeddings)

    # Print distribution
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    logging.info(f"Cluster distribution: {dist}")

    return labels


def create_output(df: pd.DataFrame, texts: list[str], coords: np.ndarray,
                  clusters: np.ndarray, categorical_fields: list,
                  config: dict) -> list[dict]:
    """Create output JSON structure."""
    output = []

    for idx, row in df.iterrows():
        point = {
            'id': str(row['id']),
            'text': texts[idx],
            'x': float(coords[idx, 0]),
            'y': float(coords[idx, 1]),
            'cluster': int(clusters[idx]),
            'nickname': row['nickname'] if row['nickname'] else 'anonymous'
        }

        # Add categorical fields
        for field in categorical_fields:
            # Convert field name to camelCase for JSON
            key = ''.join(word.title() if i > 0 else word
                         for i, word in enumerate(field.split('_')[1:]))
            if not key:
                key = field
            point[key] = row[field]

        output.append(point)

    return output


def print_summary(output: list, config: dict):
    """Print processing summary."""
    emb_config = config.get("embedding", {})

    logging.info("=" * 60)
    logging.info("PROCESSING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Embedding model: {emb_config.get('type')} / {emb_config.get('model')}")
    logging.info(f"Total points: {len(output)}")
    logging.info(f"Clusters: {config['clustering']['n_clusters']}")

    # Cluster breakdown
    clusters = {}
    for p in output:
        c = p['cluster']
        clusters[c] = clusters.get(c, 0) + 1
    logging.info(f"Distribution: {clusters}")


def main():
    parser = argparse.ArgumentParser(
        description='Hidden Connections - ML Pipeline with CLIP/LLM Support'
    )
    parser.add_argument('--config', type=Path, default=Path('pipeline/config.yaml'),
                        help='Path to config file')
    parser.add_argument('--input', type=Path, help='Override input CSV path')
    parser.add_argument('--output', type=Path, help='Override output JSON path')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    setup_logging(config)

    # Resolve paths
    base_path = Path(__file__).parent.parent
    input_path = args.input or base_path / config['paths']['input']
    output_path = args.output or base_path / config['paths']['output']

    # Load data
    df = load_data(
        input_path,
        config['text_fields'],
        config.get('categorical_fields', []),
        config.get('max_chars_per_question', 500)
    )

    # Create combined text
    field_labels = config.get('field_labels', {})
    texts = create_combined_text(df, config['text_fields'], field_labels)

    # Generate embeddings
    logging.info("Initializing embedding model...")
    backend = create_embedding_backend(config)
    logging.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = backend.encode(texts, show_progress=config.get('logging', {}).get('show_progress', True))
    logging.info(f"Embedding shape: {embeddings.shape}")

    # Reduce dimensions
    coords_2d = reduce_dimensions(embeddings, config)

    # Normalize coordinates
    coords_normalized = normalize_coordinates(coords_2d)

    # Cluster on original embeddings
    clusters = cluster_points(embeddings, config)

    # Create output
    output = create_output(
        df, texts, coords_normalized, clusters,
        config.get('categorical_fields', []), config
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = config.get('output', {}).get('indent', 2)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=indent)
    logging.info(f"Output written to: {output_path}")

    # Copy to web directory
    web_output = base_path / config['paths'].get('web_output', 'web/points.json')
    web_output.parent.mkdir(parents=True, exist_ok=True)
    with open(web_output, 'w') as f:
        json.dump(output, f, indent=indent)
    logging.info(f"Copied to: {web_output}")

    # Print summary
    print_summary(output, config)


if __name__ == '__main__':
    main()
