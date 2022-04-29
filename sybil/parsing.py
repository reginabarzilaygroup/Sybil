import argparse
import torch
import os
import pwd
from pytorch_lightning import Trainer

EMPTY_NAME_ERR = 'Name of augmentation or one of its arguments cant be empty\n\
                  Use "name/arg1=value/arg2=value" format'
POSS_VAL_NOT_LIST = (
    "Flag {} has an invalid list of values: {}. Length of list must be >=1"
)


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
        arguments = t.split("/")
        name = arguments[0]
        if name == "":
            raise Exception(EMPTY_NAME_ERR)

        kwargs = {}
        if len(arguments) > 1:
            for a in arguments[1:]:
                splited = a.split("=")
                var = splited[0]
                val = splited[1] if len(splited) > 1 else None
                if var == "":
                    raise Exception(EMPTY_NAME_ERR)

                kwargs[var] = val

        augmentations.append((name, kwargs))

    return augmentations


def parse_dispatcher_config(config):
    """
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of flag strings, each of which encapsulates one job.
        *Example: --train --cuda --dropout=0.1 ...
    returns: experiment_axies - axies that the grid search is searching over
    """
    jobs = [""]
    experiment_axies = []
    search_spaces = config["search_space"]

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
                        new_job_str = "{} --{} {}".format(
                            parent_job, flag, val_list_str
                        )
                    else:
                        new_job_str = "{} --{} {}".format(parent_job, flag, value)
                    children.append(new_job_str)
            jobs = children

    return jobs, experiment_axies


def parse_args(args_strings=None):
    parser = argparse.ArgumentParser(
        description="Sandstone research repo. Support Mammograms, CT Scans, Thermal Imaging, Cell Imaging and Chemistry."
    )
    # setup
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Whether or not to train model",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Whether or not to run model on test set",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Whether or not to run model on dev set",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        default=False,
        help="Whether or not to fine_tune model",
    )
    parser.add_argument(
        "--num_epochs_fine_tune",
        type=int,
        default=1,
        help="Num epochs to finetune model",
    )

    # data
    parser.add_argument(
        "--dataset",
        default="nlst",
        choices=[
            "sybil",
            "nlst",
            "nlst_risk_factors",
            "nlst_for_plco2012",
            "nlst_for_plco2019" "mgh",
        ],
        help="Name of dataset from dataset factory to use [default: nlst]",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Width and height of image in pixels. [default: [256,256]]",
    )
    parser.add_argument(
        "--num_chan", type=int, default=3, help="Number of channels for input image"
    )
    parser.add_argument(
        "--img_mean",
        type=float,
        nargs="+",
        default=[128.1722],
        help="Mean of image per channel",
    )
    parser.add_argument(
        "--img_std",
        type=float,
        nargs="+",
        default=[87.1849],
        help="Standard deviation  of image per channel",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/data/rsg/mammogram/NLST/nlst-ct-png",
        help="Dir of images. Note, image path in dataset jsons should stem from here",
    )
    parser.add_argument(
        "--img_file_type",
        type=str,
        default="png",
        choices=["png", "dicom"],
        help="Type of image. one of [png, dicom]",
    )
    parser.add_argument(
        "--fix_seed_for_multi_image_augmentations",
        action="store_true",
        default=False,
        help="Use same seed for each slice of volume augmentations",
    )
    parser.add_argument(
        "--dataset_file_path",
        type=str,
        default="/Mounts/rbg-storage1/datasets/NLST/full_nlst_google.json",
        help="Path to dataset file either as json or csv",
    )
    parser.add_argument(
        "--num_classes", type=int, default=6, help="Number of classes to predict"
    )

    # Alternative training/testing schemes
    parser.add_argument(
        "--cross_val_seed",
        type=int,
        default=0,
        help="Seed used to generate the partition.",
    )
    parser.add_argument(
        "--assign_splits",
        action="store_true",
        default=False,
        help="Whether to assign different splits than those predetermined in dataset",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="random",
        choices=["random", "institution_split"],
        help="How to split dataset if assign_split = True. Usage: ['random', 'institution_split'].",
    )
    parser.add_argument(
        "--split_probs",
        type=float,
        nargs="+",
        default=[0.6, 0.2, 0.2],
        help="Split probs for datasets without fixed train dev test. ",
    )

    # survival analysis setup
    parser.add_argument(
        "--max_followup", type=int, default=6, help="Max followup to predict over"
    )

    # risk factors
    parser.add_argument(
        "--use_risk_factors",
        action="store_true",
        default=False,
        help="Whether to feed risk factors into last FC of model.",
    )  #
    parser.add_argument(
        "--risk_factor_keys",
        nargs="*",
        default=[],
        help="List of risk factors to include in risk factor vector.",
    )

    # handling CT slices
    parser.add_argument(
        "--resample_pixel_spacing_prob",
        type=float,
        default=1,
        help="Probability of resampling pixel spacing into fixed dimensions. 1 when eval and using resampling",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=200,
        help="In multi image setting, the number of images per single sample.",
    )
    parser.add_argument(
        "--min_num_images",
        type=int,
        default=0,
        help="In multi image setting, the min number of images per single sample.",
    )
    parser.add_argument(
        "--slice_thickness_filter",
        type=float,
        help="Slice thickness using, if restricting to specific thickness value.",
    )
    parser.add_argument(
        "--use_only_thin_cuts_for_ct",
        action="store_true",
        default=False,
        help="Wether to use image series with thinnest cuts only.",
    )

    # region annotations
    parser.add_argument(
        "--use_annotations",
        action="store_true",
        default=False,
        help="whether to use image annotations (pixel labels) in modeling",
    )

    parser.add_argument(
        "--region_annotations_filepath", type=str, help="Path to annotations file"
    )
    parser.add_argument(
        "--annotation_loss_lambda",
        type=float,
        default=1,
        help="Weight of annotation losses",
    )
    parser.add_argument(
        "--image_attention_loss_lambda",
        type=float,
        default=1,
        help="Weight of loss for predicting image attention scores",
    )
    parser.add_argument(
        "--volume_attention_loss_lambda",
        type=float,
        default=1,
        help="Weight of loss for predicting volume attention scores",
    )

    # regularization
    parser.add_argument(
        "--primary_loss_lambda",
        type=float,
        default=1.0,
        help="Lambda to weigh the primary loss.",
    )
    parser.add_argument(
        "--adv_loss_lambda",
        type=float,
        default=1.0,
        help="Lambda to weigh the adversary loss.",
    )

    # learning
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training [default: 128]",
    )
    parser.add_argument(
        "--init_lr",
        type=float,
        default=0.001,
        help="Initial learning rate [default: 0.001]",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Amount of dropout to apply on last hidden layer [default: 0.25]",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to use [default: adam]"
    )
    parser.add_argument(
        "--momentum", type=float, default=0, help="Momentum to use with SGD"
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.1,
        help="Initial learning rate [default: 0.5]",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="L2 Regularization penaty [default: 0]",
    )
    parser.add_argument(
        "--adv_lr",
        type=float,
        default=0.001,
        help="Initial learning rate for adversary model [default: 0.001]",
    )

    # schedule
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs without improvement on dev before halving learning rate and reloading best model [default: 5]",
    )
    parser.add_argument(
        "--num_adv_steps",
        type=int,
        default=1,
        help="Number of steps for domain adaptation discriminator per one step of encoding model [default: 5]",
    )
    parser.add_argument(
        "--tuning_metric",
        type=str,
        default="c_index",
        help="Criterion based on which model is saved [default: c_index]",
    )

    # model checkpointing
    parser.add_argument(
        "--turn_off_checkpointing",
        action="store_true",
        default=False,
        help="Do not save best model",
    )

    parser.add_argument(
        "--save_dir", type=str, default="snapshot", help="Where to dump the model"
    )

    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Filename of model snapshot to load[default: None]",
    )

    # system
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Num workers for each data loader [default: 4]",
    )

    # storing results
    parser.add_argument(
        "--store_hiddens",
        action="store_true",
        default=False,
        help="Save hidden repr from each image to an npz based off results path, git hash and exam name",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=False,
        help="Save hidden repr from each image to an npz based off results path, git hash and exam name",
    )
    parser.add_argument(
        "--hiddens_dir",
        type=str,
        default="hiddens/test_run",
        help='Dir to store hiddens npy"s when store_hiddens is true',
    )
    parser.add_argument(
        "--save_attention_scores",
        action="store_true",
        default=False,
        help="Whether to save attention scores when using attention mechanism",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="logs/test.args",
        help="Where to save the result logs",
    )

    # cache
    parser.add_argument(
        "--cache_path", type=str, default=None, help="Dir to cache images."
    )
    parser.add_argument(
        "--cache_full_img",
        action="store_true",
        default=False,
        help="Cache full image locally as well as cachable transforms",
    )

    # run
    parser = Trainer.add_argparse_args(parser)
    if args_strings is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_strings)
    args.lr = args.init_lr

    if (isinstance(args.gpus, str) and len(args.gpus.split(",")) > 1) or (
        isinstance(args.gpus, int) and args.gpus > 1
    ):
        args.accelerator = "ddp"
        args.replace_sampler_ddp = False
    else:
        args.accelerator = None
        args.replace_sampler_ddp = False

    args.unix_username = pwd.getpwuid(os.getuid())[0]

    # learning initial state
    args.step_indx = 1

    return args
