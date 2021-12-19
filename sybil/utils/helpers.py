from sybil.datasets.sybil import SybilDataset
from sybil.datasets.nlst import *

SUPPORTED_DATASETS = {
    'sybil': SybilDataset,
    'nlst': NLST_Survival_Dataset,
    'nlst_risk_factors': NLST_Risk_Factor_Task,
    'nlst_for_plco': NLST_for_PLCO
}

def get_dataset(dataset_name, split, args):
    if dataset_name not in SUPPORTED_DATASETS:
        raise NotImplementedError("Dataset {} does not exist.".format(dataset_name))
    return SUPPORTED_DATASETS[dataset_name](args, split)