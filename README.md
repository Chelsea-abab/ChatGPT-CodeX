# ChatGPT-CodeX

This repository provides utilities for generating embeddings and training a
simple MLP classifier on those embeddings. Use `get_embedding_from_raw` in
`doubao_embedding.py` to obtain embedding vectors for your data. The
`dataset_processor.py` helper reads a JSONL file with the fields `item_id`,
`label`, `query_text`, and `query_image_paths`, computes embeddings, and splits
them into train/validation sets. Train the classifier with
`embedding_classifier.py`, which reports training/validation loss, AUC, F1, and
accuracy during training.

## Handling imbalanced labels

For datasets with long-tail distributions, the training utility now supports
class-aware strategies:

- **Weighted loss** – the cross entropy loss can weight each class inversely to
  its frequency so that rare labels have a stronger influence.
- **Weighted sampling** – the dataloader may use a
  `WeightedRandomSampler` to oversample minority classes during each epoch.

These behaviours are enabled by default in `train_mlp_classifier` and can be
disabled with the `use_class_weights` or `use_weighted_sampler` arguments.
