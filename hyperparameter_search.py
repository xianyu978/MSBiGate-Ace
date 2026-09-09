
#!/usr/bin/env python
from pathlib import Path
import argparse, itertools, json, random, sys, csv, shutil, torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from msbigate_ace.data import load_species_splits
from msbigate_ace.training import train_model


def save_rows(rows,path):
    if not rows:return
    cols=[]
    for r in rows:
        for k in r:
            if k not in cols:cols.append(k)
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description='Species-specific random/grid hyperparameter search matching the uploaded MSBiGate-Ace scripts.')
    p.add_argument('--species',required=True); p.add_argument('--mode',choices=['random','grid'],default='random'); p.add_argument('--max-trials',type=int,default=None)
    p.add_argument('--data-root',type=Path,default=ROOT/'data'); p.add_argument('--output-root',type=Path,default=ROOT/'outputs'/'search')
    p.add_argument('--gpu',type=int,default=0); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); spaces=json.loads((ROOT/'configs'/'search_spaces.json').read_text())
    if a.species not in spaces: raise SystemExit(f'Unknown species: {a.species}')
    entry=spaces[a.species]; space=entry['space']; keys=list(space); combos=[dict(zip(keys,v)) for v in itertools.product(*(space[k] for k in keys))]
    if a.mode=='random': random.Random(a.seed).shuffle(combos)
    max_trials=a.max_trials if a.max_trials is not None else entry['default_max_trials']; combos=combos if max_trials is None else combos[:max_trials]
    splits=load_species_splits(a.data_root,a.species); outroot=a.output_root/a.species; outroot.mkdir(parents=True,exist_ok=True); rows=[]
    for i,h in enumerate(combos,1):
        cfg={'display_name':entry['display_name'],'input_dim':43,'lstm_layers':1,'hidden_size':h['HID_BLOSUM'],'embedding_dim':h['EMB_DIM'],'cnn_channels':h['CNN_HID'],'kernels':h['CNN_KERNELS'],'batch_size':h['BATCH_SIZE'],'learning_rate':h['LR'],'weight_decay':h['WEIGHT_DECAY'],'dropout':h['DROPOUT'],'patience':h['PATIENCE'],'max_epochs':h['EPOCHS']}
        td=outroot/f'trial_{i:04d}'; print(f'\n=== trial {i}/{len(combos)}: {cfg} ===')
        try:
            r=train_model(a.species,splits,cfg,td,gpu=a.gpu,seed=a.seed,threshold_source='test')
            m=r['test_metrics']; row={'trial':i,'best_val_auc':r['best_val_auc'],'best_epoch':r['best_epoch'],'test_SN':m['SN'],'test_ACC':m['ACC'],'test_PRE':m['PRE'],'test_F1':m['F1'],'test_AUC':m['AUC'],'test_MCC':m['MCC'],'test_AUPRC':m['AUPRC'],**{k:json.dumps(v) if isinstance(v,list) else v for k,v in h.items()}}
        except Exception as e:
            row={'trial':i,'error':repr(e),**{k:json.dumps(v) if isinstance(v,list) else v for k,v in h.items()}}
        rows.append(row); save_rows(rows,outroot/'summary_all_trials_so_far.csv')
    save_rows(rows,outroot/'summary_all_trials.csv')
    valid=[r for r in rows if 'best_val_auc' in r]; valid.sort(key=lambda r:r['best_val_auc'],reverse=True); save_rows(valid[:10],outroot/'top10_by_val_AUC.csv')
    if valid:
        src=outroot/f"trial_{valid[0]['trial']:04d}"/'best_model_by_val_auc.pt'; dst=outroot/'best_overall_by_val_AUC.pt'
        if src.exists(): shutil.copy2(src,dst)
        print('Best validation-AUROC trial:',valid[0])
if __name__=='__main__': main()
