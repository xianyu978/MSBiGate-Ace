
from __future__ import annotations
from pathlib import Path
import copy, csv, json, random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from .data import make_loader
from .model import build_model, count_parameters
from .metrics import threshold_at_specificity_interp, calc_metrics


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    except Exception: pass


def select_device(gpu=0):
    if torch.cuda.is_available():
        try: torch.cuda.set_device(int(gpu)); return torch.device(f'cuda:{int(gpu)}')
        except Exception: return torch.device('cuda:0')
    return torch.device('cpu')

@torch.no_grad()
def get_scores(model,loader,device):
    model.eval(); yt=[]; yp=[]
    for x,y in loader:
        x=x.to(device); prob=torch.softmax(model(x),dim=1)[:,1]
        yt.extend(y.cpu().tolist()); yp.extend(prob.detach().cpu().tolist())
    return np.asarray(yt,dtype=np.int64),np.asarray(yp,dtype=np.float32)


def _write_csv(rows,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    cols=[]
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)


def _save_predictions(records,y_true,y_prob,path):
    rows=[]
    for i,(r,y,p) in enumerate(zip(records,y_true,y_prob)):
        rows.append({'index':i,'protein_id':r.get('protein_id',''),'k_pos':r.get('k_pos',-1),'peptide31':r.get('peptide31',''),'label':int(y),'prob_Kac':float(p)})
    _write_csv(rows,path)


def train_model(species, splits, cfg, out_dir, gpu=0, seed=42, threshold_source='test', max_epochs_override=None):
    set_seed(seed); device=select_device(gpu); out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    train_loader=make_loader(splits['train'],int(cfg['batch_size']),True,seed)
    val_loader=make_loader(splits['val'],128,False,seed); test_loader=make_loader(splits['test'],128,False,seed)
    model=build_model(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg['learning_rate']),weight_decay=float(cfg['weight_decay']))
    ys=[int(r['label']) for r in splits['train']]; n_pos=max(1,sum(ys)); n_neg=max(1,len(ys)-n_pos)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.0,(n_neg+1e-8)/(n_pos+1e-8)],device=device))
    best_auc=-1.0; best_epoch=-1; best_state=None; patience_counter=0; history=[]
    max_epochs=int(max_epochs_override or cfg['max_epochs'])
    for epoch in range(1,max_epochs+1):
        model.train(); total_loss=0.0; total_n=0; correct=0
        for x,y in train_loader:
            x=x.to(device); y=y.to(device); logits=model(x); loss=criterion(logits,y)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
            total_loss += loss.item()*y.size(0); total_n += y.size(0); correct += (logits.argmax(1)==y).sum().item()
        yv,pv=get_scores(model,val_loader,device); val_auc=roc_auc_score(yv,pv) if len(np.unique(yv))>1 else float('-inf')
        val_thr=threshold_at_specificity_interp(yv,pv,0.9); vm=calc_metrics(yv,pv,val_thr)
        history.append({'epoch':epoch,'train_loss':total_loss/max(total_n,1),'train_acc':correct/max(total_n,1),'val_AUC':float(val_auc),'val_SN':vm['SN'],'val_ACC':vm['ACC'],'val_PRE':vm['PRE'],'val_F1':vm['F1'],'val_MCC':vm['MCC'],'val_AUPRC':vm['AUPRC'],'val_SP':vm['SP'],'val_Threshold@Sp':vm['Threshold@Sp']})
        print(f'[{species}] epoch {epoch:03d} loss={history[-1]["train_loss"]:.4f} val_AUC={val_auc:.4f} val_F1@Sp0.9={vm["F1"]:.4f}')
        if val_auc > best_auc:
            best_auc=float(val_auc); best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); patience_counter=0
        else: patience_counter+=1
        if patience_counter >= int(cfg['patience']):
            print(f'[{species}] early stopping at epoch {epoch}')
            break
    if best_state is None: raise RuntimeError('No checkpoint was selected.')
    model.load_state_dict(best_state)
    ckpt=out_dir/'best_model_by_val_auc.pt'
    torch.save({'state_dict':best_state,'species':species,'config':cfg,'seed':seed,'best_epoch':best_epoch,'best_val_auc':best_auc,'sp_target':0.9,'threshold_mode':'original_interp'},ckpt)
    _write_csv(history,out_dir/'history.csv')
    yv,pv=get_scores(model,val_loader,device); yt,pt=get_scores(model,test_loader,device)
    if threshold_source=='val': threshold=threshold_at_specificity_interp(yv,pv,0.9)
    elif threshold_source=='test': threshold=threshold_at_specificity_interp(yt,pt,0.9)
    else: raise ValueError('threshold_source must be test or val')
    test_metrics=calc_metrics(yt,pt,threshold)
    _save_predictions(splits['test'],yt,pt,out_dir/'test_predictions.csv')
    result={'species':species,'device':str(device),'parameter_count':count_parameters(model),'best_epoch':best_epoch,'best_val_auc':best_auc,'threshold_source':threshold_source,'test_metrics':test_metrics,'config':cfg}
    (out_dir/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
