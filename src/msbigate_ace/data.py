
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .features import WINDOW, norm_sequence, encode_sequence_features


def parse_label(x):
    s = str(x).strip().lower()
    if s in ("1","pos","positive","yes","y","true","acetylated"): return 1
    if s in ("0","neg","negative","no","n","false","non","non-acetylated"): return 0
    return None


def load_split_file(path: Path, enforce_center_k: bool = True):
    records=[]
    with Path(path).open('r', encoding='utf-8', errors='ignore') as f:
        for row_idx,row in enumerate(csv.reader(f)):
            if not row: continue
            if len(row) >= 4:
                protein_id=(row[0] or '').strip(); pos=(row[1] or '').strip(); raw=(row[2] or '').strip(); lab=parse_label(row[3])
            elif len(row) >= 2:
                protein_id=f'{Path(path).stem}_{row_idx}'; pos='-1'; raw=(row[0] or '').strip(); lab=parse_label(row[1])
            else:
                continue
            if lab is None: continue
            seq=norm_sequence(raw, WINDOW)
            if enforce_center_k and seq[WINDOW//2] != 'K': continue
            records.append({'protein_id': protein_id or f'{Path(path).stem}_{row_idx}', 'k_pos': int(pos) if pos.isdigit() else -1, 'peptide31':seq, 'label':int(lab)})
    if not records:
        raise ValueError(f'No valid records found in {path}')
    return records


def load_species_splits(data_root: Path, species: str):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f'Data directory not found: {data_root}')
    paths = {
        'train': data_root / f'train_{species}_31.txt',
        'val': data_root / f'valid_{species}_31.txt',
        'test': data_root / f'test_{species}_31.txt',
    }
    for split_name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f'Missing {split_name} split for {species}: {path}'
            )
    return {name: load_split_file(path) for name, path in paths.items()}


class SequenceDataset(Dataset):
    def __init__(self, records):
        self.records=list(records)
        self.x=torch.tensor(np.stack([encode_sequence_features(r['peptide31']) for r in self.records]), dtype=torch.float32)
        self.y=torch.tensor(np.asarray([int(r['label']) for r in self.records], dtype=np.int64), dtype=torch.long)
    def __len__(self): return self.y.size(0)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


def make_loader(records, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0):
    ds=SequenceDataset(records)
    g=torch.Generator(); g.manual_seed(seed)
    def seed_worker(worker_id):
        import random, numpy as np, torch
        worker_seed=torch.initial_seed()%2**32
        np.random.seed(worker_seed); random.seed(worker_seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                      generator=g, drop_last=False)
