
from pathlib import Path
import sys, json, torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from msbigate_ace.features import encode_sequence_features
from msbigate_ace.model import build_model, count_parameters

def main():
    x=encode_sequence_features('A'*15+'K'+'A'*15)
    assert x.shape==(31,43)
    cfg=json.loads((ROOT/'configs/final_species_configs.json').read_text())['Arabidopsis_thaliana']
    m=build_model(cfg); y=m(torch.tensor(x[None],dtype=torch.float32))
    assert tuple(y.shape)==(1,2)
    assert count_parameters(m)==cfg['manuscript_parameters'], (count_parameters(m),cfg['manuscript_parameters'])
    print('smoke test passed')
if __name__=='__main__': main()
