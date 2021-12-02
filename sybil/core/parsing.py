import argparse
import torch
import os
import pwd
from sandstone.datasets.factory import get_dataset_class
from pytorch_lightning import Trainer

POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'

def parse_augmentations(raw_augmentations):
    """
    Parse the list of augmentations, given by configuration, into a list of
    tuple of the augmentations name and a dictionary containing additional args.

    The augmentation is assumed to be of the form 'name/arg1=value/arg2=value'

    :raw_augmentations: list of strings [unparsed augmentations]
    :returns: list of parsed augmentations [list of (name,additional_args)]

    """
    augmentations = []
    for t in raw_augmentations:
        arguments = t.split('/')
        name = arguments[0]
        if name == '':
            raise Exception(EMPTY_NAME_ERR)

        kwargs = {}
        if len(arguments) > 1:
            for a in arguments[1:]:
                splited = a.split('=')
                var = splited[0]
                val = splited[1] if len(splited) > 1 else None
                if var == '':
                    raise Exception(EMPTY_NAME_ERR)

                kwargs[var] = val

        augmentations.append((name, kwargs))

    return augmentations

def parse_dispatcher_config(config):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of flag strings, each of which encapsulates one job.
        *Example: --train --cuda --dropout=0.1 ...
    returns: experiment_axies - axies that the grid search is searching over
    '''
    jobs = [""]
    experiment_axies = []
    search_spaces = config['search_space']

    # Support a list of search spaces, convert to length one list for backward compatiblity
    if not isinstance(search_spaces, list):
        search_spaces = [search_spaces]


    for search_space in search_spaces:
        # Go through the tree of possible jobs and enumerate into a list of jobs
        for ind, flag in enumerate(search_space):
            possible_values = search_space[flag]
            if len(possible_values) > 1:
                experiment_axies.append(flag)

            children = []
            if len(possible_values) == 0 or type(possible_values) is not list:
                raise Exception(POSS_VAL_NOT_LIST.format(flag, possible_values))
            for value in possible_values:
                for parent_job in jobs:
                    if type(value) is bool:
                        if value:
                            new_job_str = "{} --{}".format(parent_job, flag)
                        else:
                            new_job_str = parent_job
                    elif type(value) is list:
                        val_list_str = " ".join([str(v) for v in value])
                        new_job_str = "{} --{} {}".format(parent_job, flag,
                                                          val_list_str)
                    else:
                        new_job_str = "{} --{} {}".format(parent_job, flag, value)
                    children.append(new_job_str)
            jobs = children

    return jobs, experiment_axies

def parse_args(args_strings=None):
    parser = argparse.ArgumentParser(description='Sandstone research repo. Support Mammograms, CT Scans, Thermal Imaging, Cell Imaging and Chemistry.')
    # setup
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--dev', action='store_true', default=False, help='Whether or not to run model on dev set')
    parser.add_argument('--fine_tune', action='store_true', default=False, help='Whether or not to fine_tune model')
    parser.add_argument('--num_epochs_fine_tune', type=int, default=1, help='Num epochs to finetune model')
    
    # lightning module
    parser.add_argument('--lightning_name', type=str, default='default', help="Name of lightning module to structure training.")

    # data
    parser.add_argument('--dataset', default='mnist', help='Name of dataset [default: nlst]')
    parser.add_argument('--img_size',  type=int, nargs='+', default=[256, 256], help='width and height of image in pixels. [default: [256,256]')
    parser.add_argument('--get_dataset_stats', action='store_true', default=False, help='Whether to compute the mean and std of the training images on the fly rather than using precomputed values')
    parser.add_argument('--metadata_file', type=str, default='/home/administrator/Mounts/Isilon/metadata', help='dir of metadata jsons.')

    # Alternative training/testing schemes
    parser.add_argument('--cross_val_seed', type=int, default=0, help="Seed used to generate the partition.")
    parser.add_argument('--assign_splits', action='store_true', default=False, help = "Whether to assign different splits than those predetermined in dataset")
    parser.add_argument('--split_type', type=str, default='random', help="How to split dataset if assign_split = True. Usage: ['random', 'institution_split'].")
    parser.add_argument('--split_probs', type=float, nargs='+', default=[0.6, 0.2, 0.2], help='Split probs for datasets without fixed train dev test. ')

    #survival analysis setup
    parser.add_argument('--max_followup', type=int, default=5, help='Max followup to predict over')
    
    # risk factors
    parser.add_argument('--use_risk_factors', action='store_true', default=False, help='Whether to feed risk factors into last FC of model.') #
    parser.add_argument('--risk_factor_keys', nargs='*', default=[], help='List of risk factors to include in risk factor vector.')

    # handling CT slices
    parser.add_argument('--max_num_images', type=int, default=200, help='In multi image setting, the number of images per single sample.')
    parser.add_argument('--min_num_images', type=int, default=0, help='In multi image setting, the min number of images per single sample.')
    parser.add_argument('--padding_method',  type=str, default='evenly', help='How to pad image series with black image. Default is evenly distrubted across slices to obtain num_images slices.')
    parser.add_argument('--truncation_method',  type=str, default='evenly', help='How to select slices if image series has more slices than num_images.')
    parser.add_argument('--slice_thickness_filter',  type=float, nargs='*', help='Slice thickness using, if restricting to specific thickness value.')
    parser.add_argument('--cross_section_filter',  type=str, nargs='*', help='Restrict to using specific cross sections [transverse, coronal, sagittal, oblique].')
    parser.add_argument('--use_only_thin_cuts', action='store_true', default=False, help='Wether to use image series with thinnest cuts only.')
    
    # region annotations
    parser.add_argument('--use_annotations', action = 'store_true', default = False, help = 'whether to use image-level annotations (pixel labels) in modeling')
    parser.add_argument('--annotation_loss_lambda', type = float, default = 1, help = 'Weight of annotation losses')

    # learning
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 128]')
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--dropout', type=float, default=0.25, help='Amount of dropout to apply on last hidden layer [default: 0.25]')
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer to use [default: adam]')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum to use with SGD')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='initial learning rate [default: 0.5]')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 Regularization penaty [default: 0]')
    
    # schedule
    parser.add_argument('--patience', type=int, default=10, help='number of epochs without improvement on dev before halving learning rate and reloading best model [default: 5]')
     
    # model checkpointing
    parser.add_argument('--save_dir', type=str, default='snapshot', help='where to dump the model')
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    
    # system 
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for each data loader [default: 4]')
        
    # cache
    parser.add_argument('--cache_path', type=str, default=None, help='dir to cache images.')

    # comet logger
    parser.add_argument('--comet_project_name', type=str, default=None, help='Name of project for comet logger')
    parser.add_argument('--comet_workspace', type=str, default=None, help='Name of workspace for comet logger')
    parser.add_argument('--comet_tags', nargs='*', default=[], help="List of tags for comet logger")

    # run
    parser = Trainer.add_argparse_args(parser)
    if args_strings is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_strings)
    args.lr = args.init_lr

    if (isinstance(args.gpus, str) and len(args.gpus.split(",")) > 1) or (isinstance(args.gpus, int) and  args.gpus > 1):
        args.distributed_backend = 'ddp'
        args.replace_sampler_ddp = False

    # Set args particular to dataset
    get_dataset_class(args).set_args(args)

    args.unix_username = pwd.getpwuid( os.getuid() )[0]

    # learning initial state
    args.step_indx = 1

    # Parse list args to appropriate data format
    parse_list_args(args)

    return args


def parse_list_args(args):
    """Converts list args to their appropriate data format.

    Includes parsing image dimension args, augmentation args,
    block layout args, and more.

    Arguments:
        args(Namespace): Config.

    Returns:
        args but with certain elements modified to be in the
        appropriate data format.
    """

    args.image_augmentations = parse_augmentations(args.image_augmentations)
    args.tensor_augmentations = parse_augmentations(args.tensor_augmentations)
    args.test_image_augmentations = parse_augmentations(args.test_image_augmentations)
    args.test_tensor_augmentations = parse_augmentations(args.test_tensor_augmentations)
    
