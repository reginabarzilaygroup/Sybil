import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from os.path import dirname, realpath
import sys
import git
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
import torch
import torch.distributed as dist
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
    repo = git.Repo(search_parent_directories=True)
    commit  = repo.head.object
    args.commit = commit.hexsha
    result_path_stem = args.results_path.split("/")[-1].split('.')[0]
    log.info("Sandstone main running from commit: \n\n{}\n{}author: {}, date: {}".format(
        commit.hexsha, commit.message, commit.author, commit.committed_date))

    if args.get_dataset_stats:
        log.info("\nComputing image mean and std...")
        args.img_mean, args.img_std = get_dataset_stats(args)
        log.info('Mean: {}'.format(args.img_mean))
        log.info('Std: {}'.format(args.img_std))

    log.info("\nLoading data-augmentation scheme...")
    augmentations = augmentation_factory.get_augmentations(
        args.image_augmentations, args.tensor_augmentations, args)
    test_augmentations = augmentation_factory.get_augmentations(
        args.test_image_augmentations, args.test_tensor_augmentations, args)
    # Load dataset and add dataset specific information to args
    log.info("\nLoading data...")
    train_data, dev_data, test_data = dataset_factory.get_dataset(args, augmentations, test_augmentations)

    model = model_factory.get_model_by_name(args.model_name, args)

    log.info("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in ['optimizer_state', 'patient_to_partition_dict', 'path_to_hidden_dict', 'exam_to_year_dict', 'exam_to_device_dict', 'treatment_to_index','drug_to_y']:
            log.info("\t{}={}".format(attr.upper(), value))

    save_path = args.results_path
    log.info("\n")

    log.info("\n")
    if args.dev or args.store_hiddens:
        log.info("-------------\nDev")
        model.save_prefix = 'dev_'
        model.test(dev_data)
    log.info("\n")
    if args.test or args.store_hiddens:
        log.info("-------------\nTest")
        model.save_prefix = 'test_'
        model.test(test_data)

    if args.store_hiddens or args.eval_train:
        log.info("---\n Now running Eval on train to store final hiddens for each train sample...")
        model.save_prefix = 'eval_train_'
        model.test(train_data)
    
    log.info("Saving args to {}".format(args.results_path))
    pickle.dump(vars(args), open(args.results_path,'wb'))

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = parsing.parse_args()
    main(args)
