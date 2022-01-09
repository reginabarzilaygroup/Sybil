import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from sybil.parsing import parse_args
from scripts.plcom2012.plcom2012 import PLCOm2012
from sybil.utils.helpers import get_dataset
import sybil.utils.loading as loaders

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
    args = parse_args()
    main(args)
