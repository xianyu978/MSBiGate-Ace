
#!/usr/bin/env python
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from msbigate_ace.data import load_species_splits
from msbigate_ace.training import train_model


def main():
    p=argparse.ArgumentParser(description='Train the final MSBiGate-Ace configuration for one or all species.')
    p.add_argument('--species',default='Arabidopsis_thaliana',help='Species key or "all".')
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs'/'final_models')
    p.add_argument('--gpu',type=int,default=0); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threshold-source',choices=['test','val'],default='test',help='Use test to reproduce the manuscript fixed-Sp comparison; val is preferred for prospective deployment.')
    p.add_argument('--max-epochs-override',type=int,default=None,help='Testing/debug only.')
    a=p.parse_args()
    cfgs=json.loads((ROOT/'configs'/'final_species_configs.json').read_text())
    species=list(cfgs) if a.species.lower()=='all' else [a.species]
    for sp in species:
        if sp not in cfgs: raise SystemExit(f'Unknown species: {sp}. Choose from: {", ".join(cfgs)}')
        splits=load_species_splits(a.data_root,sp)
        out=a.output_root/sp
        result=train_model(sp,splits,cfgs[sp],out,gpu=a.gpu,seed=a.seed,threshold_source=a.threshold_source,max_epochs_override=a.max_epochs_override)
        print(json.dumps(result,indent=2))
if __name__=='__main__': main()
