import pickle
from sklearn import metrics

file_name="test.pkl"
begin="/data/pkl_files/1"
decision_grid=[float(i)/float(100) for i in range(1,100)]#[0.5]

def calc_mean(res_folds):
    min_len=min([len(x) for x in res_folds])
    mean=[0.0] * min_len
    for i in range(len(res_folds)):
        for j in range(min_len):
            mean[j]+=res_folds[i][j]
    for j in range(len(mean)):
        mean[j]/=float(len(res_folds))
    return mean

pos_mean,neg_mean,all_f1=[],[],[]
optim_decision,best_f1_mean=[],[]

runs=[]

counted_docs=0
ids_not_insite=set()

prec_100,rec_100,fmes_100,prec,rec=[],[],[],[],[]
prec_b,rec_b,f_b=[],[],[]

def find_decision(pred,true_clss,cl):
    best_decisions=[]
    best_f1=[]
    for predictions in pred:
        f1_max=-1
        dec_bound_max=-1
        for dec_bound in decision_grid:
            predicted=[0 if p[0]>dec_bound else clss for p in predictions] if clss!=0 else [1 if p[0]>dec_bound else clss for p in predictions]
            _, _, fscore, _ = metrics.precision_recall_fscore_support(y_true=true_clss, y_pred=predicted,pos_label=cl)
            f1_mes=fscore[cl]
            #f1_mes=metrics.f1_score(true_clss,predicted)
            if f1_mes>f1_max:
                f1_max=f1_mes
                dec_bound_max=dec_bound
        best_decisions.append(dec_bound_max)
        best_f1.append(f1_max)
    return best_decisions,best_f1
import numpy as np

def calc_f(test, pred,clss):
    prec, rec,f_m=[],[],[]
    for p in pred:
        predictet=[np.argmax(x) for x in p]
        precision,recall, fscore, _ = metrics.precision_recall_fscore_support(y_true=test, y_pred=predictet)
        prec.append(precision[clss])
        rec.append(recall[clss])
        f_m.append(fscore[clss])
    return prec,rec,f_m

import random

def precision_100_rand(true_c):
    num_attemps=1000
    prec_sum=0.0
    rec_sum=0.0
    f_sum=0.0
    for _ in range(num_attemps):
        keys=random.sample(range(len(true_c)),100)
        pred=[0]*len(true_c)
        for key in keys:
            pred[key]=1
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(true_c,pred)
        prec_sum+=precision[1]
        rec_sum+=recall[1]
        f_sum+=fscore[1]
    return [prec_sum/float(num_attemps)],[rec_sum/float(num_attemps)],[f_sum/float(num_attemps)]

def calc_100(true_c, pred,clss):
    precision_100, recall_100,f_100=[],[],[]
    prec_baseline, rec_baselien, f_baseline=precision_100_rand(true_c)
    for predictions in pred:
        x={}
        for c,p in enumerate(predictions):
            x[c]=p[clss]
        sorted_by_value = sorted(x.items(), key=lambda kv: kv[1])
        pred_100=[0]*len(true_c) if clss!=0 else [1]*len(true_c)
        for i in range(100):
            pred_100[sorted_by_value[len(sorted_by_value)-i-1][0]]=clss
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(true_c,pred_100,pos_label=clss)
        precision_100.append(precision[clss])
        recall_100.append(recall[clss])
        f_100.append(fscore[clss])

    return precision_100, recall_100,f_100,prec_baseline,rec_baselien,f_baseline


for i in range(10):
    folds=[]
    pos_fold,neg_fold,decision_fold=[],[],[]
    best_f1_fold=[]

    precision_100_f, recall_100_f, f_100_f, =[],[],[]
    prec_baseline_f, rec_baseline_f, f_baseline_f=[],[],[]
    precision_f,recall_f=[],[]
    for j in range(3):
        path=begin+str(j)+str(i)+file_name
        res=open(path,'rb')
        empty=True
        for c,fine in enumerate(res):
            if c==2:
                empty=False
                break
        res.close()
        res = open(path, 'rb')
        if not empty:
            data = pickle.load(res)
            if data:
                counted_docs+=1
                clss=1
                pos_fold.append(data["num_pos"])
                neg_fold.append(data["num_neg"])

                precision,recall,ff=calc_f(data["test_labs"],data["prediction"],clss)
                folds.append(ff)
                precision_f.append(precision)
                recall_f.append(recall)

                precision_100, recall_100,f_100,prec_baseline,rec_baseline,f_baseline = calc_100(data["test_labs"], data["prediction"],clss)
                precision_100_f.append(precision_100)
                recall_100_f.append(recall_100)
                f_100_f.append(f_100)
                prec_baseline_f.append(prec_baseline)
                rec_baseline_f.append(rec_baseline)
                f_baseline_f.append(f_baseline)

                optis,best_f1=find_decision(data["prediction"],data["test_labs"],clss)
                best_f1_fold.append(best_f1)
                decision_fold.append(optis)

            else:
                ids_not_insite.add(str(j)+str(i))

    runs.append(calc_mean(folds))

    pos_mean.append(calc_mean(pos_fold))
    neg_mean.append(calc_mean(neg_fold))
    best_f1_mean.append(calc_mean(best_f1_fold))
    optim_decision.append(calc_mean(decision_fold))

    prec_100.append(calc_mean(precision_100_f))
    rec_100.append(calc_mean(recall_100_f))
    fmes_100.append(calc_mean(f_100_f))
    prec.append(calc_mean(precision_f))
    rec.append(calc_mean(recall_f))
    prec_b.append(calc_mean(prec_baseline_f))
    rec_b.append(calc_mean(rec_baseline_f))
    f_b.append(calc_mean(f_baseline_f))


print(counted_docs)
print (ids_not_insite)
print("f1",calc_mean(runs))

print("precision",calc_mean(prec))
print("recall",calc_mean(rec))
print("pos_mean",calc_mean(pos_mean))
print("neg_mean",calc_mean(neg_mean))
print ("opt_decision",calc_mean(optim_decision))
print("best_f1",calc_mean(best_f1_mean))
print("precision@100",calc_mean(prec_100))
print("recall@100",calc_mean(rec_100))
print ("fmeasure@100",calc_mean(fmes_100))
print("precision@100 bound",calc_mean(prec_b))
print("recall@100 bound",calc_mean(rec_b))
print ("fmeasure@100 bound",calc_mean(f_b))






