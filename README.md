# ChatGPT-CodeX

This repository provides utilities for generating embeddings and training a
simple MLP classifier on those embeddings. Use `get_embedding_from_raw` in
`doubao_embedding.py` to obtain embedding vectors for your data. The
`dataset_processor.py` helper reads a JSONL file with the fields `item_id`,
`label`, `query_text`, and `query_image_paths`, computes embeddings, and splits
them into train/validation sets. Train the classifier with
`embedding_classifier.py`, which reports training/validation loss, AUC, F1, and
accuracy during training.
