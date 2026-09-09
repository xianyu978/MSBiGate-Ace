Benchmark data
All predefined dataset partitions are stored directly in this directory. No species-specific subdirectories are used.
File naming convention:
```text
train_<Species>_31.txt
valid_<Species>_31.txt
test_<Species>_31.txt
```
For example:
```text
train_Bacillus_velezensis_31.txt
valid_Bacillus_velezensis_31.txt
test_Bacillus_velezensis_31.txt
```
Each row is comma-separated and normally contains:
```text
protein_id,lysine_position,31-residue_window,label
```
`label = 1` denotes a Kac site and `label = 0` denotes a non-Kac site. Sequence windows have length 31 and are centered on the candidate lysine. Padding/non-standard symbols are normalized by the feature encoder.
