# Experimental environment

Reported manuscript environment:

- GPU: NVIDIA GeForce RTX 4090
- CUDA Toolkit: 12.1
- Python: 3.8.20
- PyTorch: 2.4.1+cu121
- PyTorch Geometric: 2.6.1
- scikit-learn: 1.3.2
- NumPy: 1.24.3
- pandas: 2.0.3
- Matplotlib: 3.7.5

The released MSBiGate-Ace implementation is sequence-only and does not import `torch_geometric`. PyTorch Geometric is retained only in `requirements-full.txt` as a record of the wider server environment stated for the manuscript.
