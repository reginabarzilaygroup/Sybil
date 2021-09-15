from sandstone.learn.metrics.factory import RegisterMetric
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, average_precision_score
import numpy as np
import pdb

EPSILON = 1e-6
BINARY_CLASSIF_THRESHOLD = 0.5


@RegisterMetric("classification")
def get_accuracy_metrics(logging_dict, args):
    stats_dict = OrderedDict()

    prob_keys = [ k[:-5] for k in logging_dict.keys() if k.endswith('probs') ]
    for key in prob_keys: 
        golds = np.array(logging_dict['{}golds'.format(key)]).reshape(-1)
        probs = np.array(logging_dict['{}probs'.format(key)])
        preds = probs.argmax(axis=-1).reshape(-1)
        probs = probs.reshape( (-1, probs.shape[-1]))

        stats_dict['{}accuracy'.format(key)] = accuracy_score(y_true=golds, y_pred=preds)

        if (args.num_classes == 2) and not (np.unique(golds)[-1] > 1 or np.unique(preds)[-1] > 1):
            stats_dict['{}precision'.format(key)] = precision_score(y_true=golds, y_pred=preds)
            stats_dict['{}recall'.format(key)] = recall_score(y_true=golds, y_pred=preds)
            stats_dict['{}f1'.format(key)] = f1_score(y_true=golds, y_pred=preds)
            num_pos = golds.sum()
            if num_pos > 0 and num_pos < len(golds):
                stats_dict['{}auc'.format(key)] = roc_auc_score(golds, probs[:,-1], average='samples')
                stats_dict['{}ap_score'.format(key)] = average_precision_score(golds, probs[:,-1], average='samples')
                precision, recall, _ = precision_recall_curve(golds, probs[:,-1])
                stats_dict['{}prauc'.format(key)] = auc(recall, precision)


    if args.num_classes >100:
        sorted_pred = np.argsort(probs, axis=1, kind='mergesort')[:, ::-1]
        for k in [5,10,20,50]:
            top_k_score = (golds == sorted_pred[:, :k].T).any(axis=0).mean()
            stats_dict['top_{}_accuracy'.format(k)] = top_k_score
    
    if len(prob_keys) > 1:
        stats_dict['accuracy'] = np.mean([stats_dict['{}accuracy'.format(key)] for key in prob_keys ])

    return stats_dict
