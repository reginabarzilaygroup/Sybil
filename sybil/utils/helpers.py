from sybil.datasets.validation import CSVDataset
from sybil.datasets.nlst import *
from sybil.datasets.mgh import MGH_Dataset
from sybil.datasets.joint_nlst_mgh import MGH_NLST_Combined_Dataset

SUPPORTED_DATASETS = {
    'validation': CSVDataset,
    'nlst': NLST_Survival_Dataset,
    'nlst_risk_factors': NLST_Risk_Factor_Task,
    'nlst_for_plco': NLST_for_PLCO,
    'mgh': MGH_Dataset,
    'nlst_mgh': MGH_NLST_Combined_Dataset
}

def get_dataset(dataset_name, split, args):
    if dataset_name not in SUPPORTED_DATASETS:
        raise NotImplementedError("Dataset {} does not exist.".format(dataset_name))
    return SUPPORTED_DATASETS[dataset_name](args, split)