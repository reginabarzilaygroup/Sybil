import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from scripts.plcom2012.plcom2012 import PLCOm2012

from sybil.utils.helpers import get_dataset
import sybil.utils.losses as losses
import sybil.utils.metrics as metrics
import sybil.utils.loading as loaders
import sybil.models.sybil as model
from sybil.parsing import parse_args



import sandstone.datasets.factory as dataset_factory
import sandstone.models.factory as model_factory
import sandstone.augmentations.factory as augmentation_factory
import sandstone.utils.parsing as parsing
import warnings
from sandstone.utils.dataset_stats import get_dataset_stats
from pytorch_lightning import _logger as log
from argparse import Namespace

# NOTE: USING GLOO by editing  py38/lib/python3.8/site-packages/pytorch_lightning/accelerators/accelerator.py

#Constants
DATE_FORMAT_STR = "%Y-%m-%d:%H-%M-%S"

def main(args):
    # Load dataset and add dataset specific information to args
    print("\nLoading data...")
    test_data = loaders.get_eval_dataset_loader(
            args,
            get_dataset(args.dataset, 'test', args),
            False
            )
    
    model = PLCOm2012(args)

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in ['optimizer_state', 'patient_to_partition_dict', 'path_to_hidden_dict', 'exam_to_year_dict', 'exam_to_device_dict', 'treatment_to_index','drug_to_y']:
            print("\t{}={}".format(attr.upper(), value))

    print("-------------\nTesting on PLCOm2012")
    model.save_prefix = 'test_'
    model.test(test_data)
    
    print("Saving args to {}".format(args.results_path))
    pickle.dump(vars(args), open(args.results_path,'wb'))

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = parsing.parse_args()
    main(args)
