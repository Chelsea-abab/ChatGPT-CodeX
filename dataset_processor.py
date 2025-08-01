import json
import random
from typing import Dict, List, Tuple

import numpy as np

from doubao_embedding import get_embedding_from_raw


def load_embeddings_from_jsonl(
    jsonl_path: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[
    List[np.ndarray],
    List[int],
    List[np.ndarray],
    List[int],
    Dict[str, int],
]:
    """Load embeddings and labels from a JSONL file.

    Each line in the JSONL file should contain the keys:
    ``item_id``, ``label``, ``query_text``, ``query_image_paths``.
    Embeddings are computed with :func:`get_embedding_from_raw`.
    The dataset is randomly split into training and validation sets.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.Random(seed).shuffle(data)

    label2id: Dict[str, int] = {}
    embeddings: List[np.ndarray] = []
    labels: List[int] = []

    for entry in data:
        item_id = entry.get("item_id")
        label = entry.get("label")
        query_text = entry.get("query_text")
        query_image_paths = entry.get("query_image_paths") or []

        emb = get_embedding_from_raw(item_id, query_text, query_image_paths)
        if emb is None:
            continue

        if label not in label2id:
            label2id[label] = len(label2id)

        embeddings.append(emb)
        labels.append(label2id[label])

    split_idx = int(len(embeddings) * (1 - val_ratio))
    train_embeddings = embeddings[:split_idx]
    train_labels = labels[:split_idx]
    val_embeddings = embeddings[split_idx:]
    val_labels = labels[split_idx:]

    return train_embeddings, train_labels, val_embeddings, val_labels, label2id
