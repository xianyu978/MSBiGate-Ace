
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, brier_score_loss

def threshold_at_specificity_interp(y_true,y_prob,target_sp=0.9):
    y_true=np.asarray(y_true); y_prob=np.asarray(y_prob)
    if len(np.unique(y_true))<2: return 0.5
    fpr,tpr,thr=roc_curve(y_true,y_prob); sp=1-fpr
    order=np.argsort(sp); sp=sp[order]; thr=thr[order]
    i=np.searchsorted(sp,target_sp)
    if i<=0: T=float(thr[0])
    elif i>=len(sp): T=float(thr[-1])
    else:
        sp0,sp1=sp[i-1],sp[i]; th0,th1=thr[i-1],thr[i]
        T=float((th0+th1)/2.0) if sp1==sp0 else float(th0+(target_sp-sp0)/(sp1-sp0)*(th1-th0))
    if np.isinf(T): T=float(np.max(y_prob)+1e-6)
    return T

def expected_calibration_error(y_true,y_prob,n_bins=10):
    y_true=np.asarray(y_true,dtype=np.float32); y_prob=np.clip(np.asarray(y_prob,dtype=np.float32),0,1)
    if len(y_true)==0: return float('nan')
    edges=np.linspace(0,1,n_bins+1); ece=0.0; n=len(y_true)
    for i in range(n_bins):
        left,right=edges[i],edges[i+1]
        mask=(y_prob>=left)&(y_prob<=right) if i==n_bins-1 else (y_prob>=left)&(y_prob<right)
        if not mask.any(): continue
        ece += (mask.sum()/n)*abs(float(y_prob[mask].mean())-float(y_true[mask].mean()))
    return float(ece)

def calc_metrics(y_true,y_prob,threshold):
    y_true=np.asarray(y_true,dtype=np.int64); y_prob=np.asarray(y_prob,dtype=np.float32)
    pred=(y_prob>=threshold).astype(np.int64)
    tn,fp,fn,tp=confusion_matrix(y_true,pred,labels=[0,1]).ravel()
    return {
        'SN':float(tp/(tp+fn+1e-8)), 'SP':float(tn/(tn+fp+1e-8)),
        'ACC':float(accuracy_score(y_true,pred)), 'PRE':float(precision_score(y_true,pred,zero_division=0)),
        'F1':float(f1_score(y_true,pred,zero_division=0)),
        'MCC':float(matthews_corrcoef(y_true,pred) if len(np.unique(pred))>1 else 0.0),
        'AUC':float(roc_auc_score(y_true,y_prob) if len(np.unique(y_true))>1 else float('nan')),
        'AUPRC':float(average_precision_score(y_true,y_prob) if len(np.unique(y_true))>1 else float('nan')),
        'Brier':float(brier_score_loss(y_true,y_prob) if len(np.unique(y_true))>1 else float('nan')),
        'ECE':expected_calibration_error(y_true,y_prob,10), 'Threshold@Sp':float(threshold),
        'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn),
    }
