# MSBiGate-Ace

Official code and reproducibility package for the manuscript:

**MSBiGate-Ace: a center-aware multi-scale CNN-BiLSTM gated fusion framework for species-specific lysine acetylation-site prediction**

MSBiGate-Ace predicts species-specific lysine acetylation (Kac) sites from 31-residue windows centered on candidate lysines. Each residue is represented by BLOSUM62 scores (20 dimensions), a 21-dimensional amino-acid one-hot vector, normalized relative position, and a center-position flag, producing a 31 × 43 input matrix. A multi-scale CNN branch and a bidirectional LSTM branch learn complementary local and contextual representations, which are integrated by an adaptive gate.

## Repository structure

```text
MSBiGate-Ace/
├── train.py                        # train manuscript-selected configurations
├── hyperparameter_search.py        # species-specific hyperparameter search
├── evaluate.py                     # evaluate a saved checkpoint
├── configs/
│   ├── final_species_configs.json  # final architecture/training settings
│   └── search_spaces.json          # species-specific search spaces
├── data/                            # all 27 train/valid/test files in one directory
├── src/msbigate_ace/               # reusable model/data/metrics/training code
├── reference/                       # dataset counts and manuscript reference results
├── docs/                            # environment and reproducibility notes
├── tests/                           # lightweight smoke test
├── requirements.txt
├── requirements-full.txt
├── environment.yml
└── LICENSE
```

The public package intentionally uses **one shared implementation** rather than nine nearly identical species-specific Python scripts. Species-specific differences are stored in the configuration files under `configs/`.

## Model

For each 31-residue sequence window, the model uses:

- BLOSUM62 substitution scores: 31 × 20
- amino-acid one-hot encoding: 31 × 21
- normalized relative position: 31 × 1
- center flag: 31 × 1
- concatenated input: 31 × 43
- LayerNorm + linear projection + GELU + dropout
- one-layer bidirectional LSTM with center-state and mean readout
- multi-scale 1D CNN with center and global-max readout
- adaptive gated fusion of the BiLSTM and CNN representations
- MLP classifier: 2h → 128 → 2

The nine species are modeled separately using species-specific configurations.

## Experimental environment

Experiments reported in the manuscript were conducted on a server equipped with an **NVIDIA GeForce RTX 4090 GPU** and **CUDA Toolkit 12.1**. The reported software environment was:

- Python 3.8.20
- PyTorch 2.4.1 + cu121
- PyTorch Geometric 2.6.1
- scikit-learn 1.3.2
- NumPy 1.24.3
- pandas 2.0.3
- Matplotlib 3.7.5

The released MSBiGate-Ace model code itself is sequence-based and does not import PyTorch Geometric. `requirements.txt` therefore contains the dependencies used directly by the released implementation, whereas `requirements-full.txt` records the wider software environment reported for the experiments.

## Installation

```bash
git clone https://github.com/xingleyun/MSBiGate-Ace.git
cd MSBiGate-Ace
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install --upgrade pip
```

For the CUDA 12.1 PyTorch build used in the experiments:

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data

All predefined train/validation/test partitions are stored **directly under `data/`**, without separate species subdirectories.

The nine species are:

- `Rattus_norvegicus`
- `Schistosoma_japonicum`
- `Saccharomyces_cerevisiae`
- `Mus_musculus`
- `Escherichia_coli`
- `Bacillus_velezensis`
- `Plasmodium_falciparum`
- `Oryza_sativa`
- `Arabidopsis_thaliana`

Naming convention:

```text
data/train_<Species>_31.txt
data/valid_<Species>_31.txt
data/test_<Species>_31.txt
```

Rows are comma-separated and normally contain:

```text
protein_id,lysine_position,31-residue_window,label
```

`label = 1` denotes Kac and `label = 0` denotes non-Kac. Exact split counts are provided in `reference/dataset_counts.csv`.

## Train a manuscript-selected configuration

Example for *B. velezensis*:

```bash
python train.py --species Bacillus_velezensis --gpu 0
```

Run all nine species sequentially:

```bash
python train.py --species all --gpu 0
```

Outputs are written to `outputs/final_models/<species>/` and include the validation-selected checkpoint, training history, predictions, and metrics.

## Hyperparameter search

Example:

```bash
python hyperparameter_search.py \
  --species Bacillus_velezensis \
  --mode random \
  --max-trials 500 \
  --gpu 0
```

The nine species-specific search spaces are stored in `configs/search_spaces.json`. A single generic search script is used for every species.

## Evaluate a trained checkpoint

```bash
python evaluate.py \
  --checkpoint outputs/final_models/Bacillus_velezensis/best_model_by_val_auc.pt \
  --species Bacillus_velezensis \
  --threshold-source test
```

## Evaluation protocol

Checkpoint selection is based on **validation AUROC**. For the manuscript's primary fixed-specificity comparison, probabilities are then obtained on the independent test split and the ROC-derived operating threshold is determined to target **Sp = 0.900**. Sn, Acc, Precision, F1, and MCC are calculated at this operating point; AUROC and AUPRC are calculated from continuous prediction scores.

To reproduce the manuscript fixed-specificity protocol:

```bash
python train.py --species Rattus_norvegicus --threshold-source test
```

For prospective use on truly unseen data, a validation-derived threshold is more appropriate:

```bash
python train.py --species Rattus_norvegicus --threshold-source val
```

## Final configurations

The manuscript-selected architecture and training settings for all nine species are stored in:

```text
configs/final_species_configs.json
```

This avoids maintaining nine duplicate training scripts while preserving every species-specific parameter choice.

## Pretrained model weights

Pretrained `.pt` checkpoints are **not included in this release**. The repository contains source code, benchmark data, final species-specific configurations, and hyperparameter-search spaces so that all nine models can be retrained from the supplied partitions.

## Reproducibility notes

- random seed: 42
- optimizer: AdamW
- loss: class-weighted cross-entropy
- positive-class weight: `N_neg / N_pos` from the training split
- gradient clipping: maximum norm 5.0
- checkpoint selection: highest validation AUROC
- early stopping: species-specific patience
- input window length: 31
- candidate lysine position: center residue

See `docs/REPRODUCIBILITY.md` for additional details.

## Citation

If you use MSBiGate-Ace, please cite the associated manuscript. Journal, DOI, and final bibliographic information can be added after publication.

## License

The source code is released under the MIT License. The benchmark datasets originate from public resources and previously published benchmark construction; dataset reuse remains subject to the attribution and reuse conditions of the original sources cited in the manuscript.
