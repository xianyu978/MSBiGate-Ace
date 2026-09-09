# Reproducibility protocol

1. Use the predefined train/validation/test partitions supplied in `data/`.
2. Encode each residue with BLOSUM62 (20), one-hot identity (21), normalized relative position (1), and center flag (1), giving 43 dimensions per residue.
3. Train a species-specific model with the configuration in `configs/final_species_configs.json`.
4. Optimize with AdamW and class-weighted cross-entropy. The positive-class weight is calculated from the training split as N_neg/N_pos.
5. Clip gradient norm at 5.0 and use random seed 42.
6. After every epoch, compute validation AUROC. Retain the checkpoint with the highest validation AUROC and stop after the configured patience without improvement.
7. For the manuscript's primary fixed-specificity comparison, obtain probabilities on the test split and determine the ROC-derived threshold targeting Sp=0.900 using the interpolation routine implemented in `metrics.py`. Then calculate Sn, Acc, Precision, F1, and MCC.
8. AUROC and AUPRC are calculated directly from continuous test probabilities.

The `expected_selected_epoch` field in the final configuration file records the checkpoint epoch reported in Supplementary Table S1. It is a reference value, not a hard-coded training stop.

For prospective prediction on truly unseen data, do not derive a threshold from that unseen test set. Use a threshold fixed on validation data (`--threshold-source val`) or a separately pre-specified operating threshold.
