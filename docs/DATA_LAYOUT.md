# Data layout

All nine species share one flat `data/` directory. Each species has three predefined partitions:

```text
data/
├── train_<Species>_31.txt
├── valid_<Species>_31.txt
└── test_<Species>_31.txt
```

There are 27 split files in total (9 species × 3 partitions).

Rows are comma-separated and normally contain four fields:

```text
protein_id,lysine_position,sequence_window,label
```

The sequence window has length 31 and the candidate lysine is position 16 (zero-based index 15). Padding/non-standard symbols such as `*` are normalized to `X`. The loader rejects windows whose center is not K.
