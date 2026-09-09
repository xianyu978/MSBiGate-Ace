
#!/usr/bin/env python
from pathlib import Path
import argparse, json, sys, torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from msbigate_ace.data import load_species_splits, make_loader
from msbigate_ace.model import build_model
from msbigate_ace.training import select_device, get_scores
from msbigate_ace.metrics import threshold_at_specificity_interp, calc_metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--species',required=True)
    p.add_argument('--data-root',type=Path,default=ROOT/'data'); p.add_argument('--gpu',type=int,default=0); p.add_argument('--threshold-source',choices=['test','val'],default='test')
    a=p.parse_args(); device=select_device(a.gpu); obj=torch.load(a.checkpoint,map_location=device); cfg=obj['config']
    model=build_model(cfg).to(device); model.load_state_dict(obj['state_dict']); splits=load_species_splits(a.data_root,a.species)
    vl=make_loader(splits['val'],128,False,42); tl=make_loader(splits['test'],128,False,42)
    yv,pv=get_scores(model,vl,device); yt,pt=get_scores(model,tl,device)
    thr=threshold_at_specificity_interp(yv,pv,0.9) if a.threshold_source=='val' else threshold_at_specificity_interp(yt,pt,0.9)
    print(json.dumps(calc_metrics(yt,pt,thr),indent=2))
if __name__=='__main__': main()
