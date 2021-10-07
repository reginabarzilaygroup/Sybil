import argparse
import torch
import os
import pwd
from sandstone.datasets.factory import get_dataset_class
from pytorch_lightning import Trainer

EMPTY_NAME_ERR = 'Name of augmentation or one of its arguments cant be empty\n\
                  Use "name/arg1=value/arg2=value" format'
BATCH_SIZE_SPLIT_ERR = 'batch_size (={}) should be a multiple of batch_splits (={})'
INVALID_IMG_TRANSFORMER_SPEC_ERR = 'Invalid image transformer embedding args. Must be length 3, as [name/size=value/dim=value]. Received {}'
INVALID_IMG_TRANSFORMER_EMBED_SIZE_ERR = 'Image transformer embeddings have different embedding dimensions {}'
INVALID_NUM_BLOCKS_ERR = 'Invalid block_layout. Must be length 4. Received {}'
INVALID_BLOCK_SPEC_ERR = 'Invalid block specification. Must be length 2 with "block_name,num_repeats". Received {}'
POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'
INVALID_DATASET_FOR_SURVIVAL = "A dataset with '_full_future'  can only be used with survival_analysis_setup and viceversa."
NPZ_MULTI_IMG_ERROR = "Npz loading code assumes multi images are in one npz and code is only in multi-img code flow."
SELF_SUPER_ERROR = "Moco and Byol only supported with instance disrimination task. Must be multi image with 2 images"

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

def parse_embeddings(raw_embeddings):
    """
    Parse the list of embeddings, given by configuration, into a list of
    tuple of the embedding embedding_name, size ('vocab size'), and the embedding dimension.

    :raw_embeddings: list of strings [unparsed transformers], each of the form 'embedding_name/size=value/dim=value'
    :returns: list of parsed embedding objects [(embedding_name, size, dim)]

    For example:
        --hidden_transformer_embeddings time_seq/size=10/dim=32 view_seq/size=2/dim=32 side_seq/size=2/dim=32
    returns
        [('time_seq', 10, 32), ('view_seq', 2, 32), ('side_seq', 2, 32)]
    """
    embeddings = []
    for t in raw_embeddings:
        arguments = t.split('/')
        if len(arguments) != 3:
                raise Exception(INVALID_IMG_TRANSFORMER_SPEC_ERR.format(len(arguments)))
        name = arguments[0]
        size = arguments[1].split('=')[-1]
        dim = arguments[2].split('=')[-1]

        embeddings.append((name, int(size), int(dim)))

    if not all([embed[-1] == int(dim) for embed in embeddings]):
        raise Exception(INVALID_IMG_TRANSFORMER_EMBED_SIZE_ERR.format([embed[-1] for embed in embeddings]))
    return embeddings

def validate_raw_block_layout(raw_block_layout):
    """Confirms that a raw block layout is in the right format.

    Arguments:
        raw_block_layout(list): A list of strings where each string
            is a layer layout in the format
            'block_name,num_repeats-block_name,num_repeats-...'

    Raises:
        Exception if the raw block layout is formatted incorrectly.
    """

    # Confirm that each layer is a list of block specifications where
    # each block specification has length 2 (i.e. block_name,num_repeats)
    for raw_layer_layout in raw_block_layout:
        for raw_block_spec in raw_layer_layout.split('-'):
            if len(raw_block_spec.split(',')) != 2:
                raise Exception(INVALID_BLOCK_SPEC_ERR.format(raw_block_spec))


def parse_block_layout(raw_block_layout):
    """Parses a ResNet block layout, which is a list of layer layouts
    with each layer layout in the form 'block_name,num_repeats-block_name,num_repeats-...'

    Example:
        ['BasicBlock,2',
         'BasicBlock,1-NonLocalBlock,1',
         'BasicBlock,3-NonLocalBlock,2-Bottleneck,2',
         'BasicBlock,2']
        ==>
        [[('BasicBlock', 2)],
         [('BasicBlock', 1), ('NonLocalBlock', 1)],
         [('BasicBlock', 3), ('NonLocalBlock', 2), ('Bottleneck', 2)],
         [('BasicBlock', 2)]]

    Arguments:
        raw_block_layout(list): A list of strings where each string
            is a layer layout as described above.

    Returns:
        A list of lists of length 4 (one for each layer of ResNet). Each inner list is
        a list of tuples, where each tuple is (block_name, num_repeats).
    """

    validate_raw_block_layout(raw_block_layout)

    block_layout = []
    for raw_layer_layout in raw_block_layout:
        raw_block_specs = raw_layer_layout.split('-')
        layer = [raw_block_spec.split(',') for raw_block_spec in raw_block_specs]
        layer = [(block_name, int(num_repeats)) for block_name, num_repeats in layer]
        block_layout.append(layer)

    return block_layout


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
    parser.add_argument('--eval_train', action='store_true', default=False, help='Whether or not to evaluate model on train set')
    parser.add_argument('--fine_tune', action='store_true', default=False, help='Whether or not to fine_tune model')
    parser.add_argument('--num_epochs_fine_tune', type=int, default=1, help='Num epochs to finetune model')
    parser.add_argument('--lightning_name', type=str, default='default', help="Name of lightning module to structure training.")
    parser.add_argument('--debug', action='store_true', default=False, help='Set sandstone to debug mode. Load only 1000 rows in metadata, set num workers to 0, max train and dev small.')
    parser.add_argument('--num_steps_alt_optimization', type=int, default=0, help='Number of steps to train alt model per training main model.')
    parser.add_argument('--eval_only', action='store_true', default=False, help='Whether only doing an evaluation')

    # data
    parser.add_argument('--dataset', default='mnist', help='Name of dataset from dataset factory to use [default: mnist]')
    parser.add_argument('--scramble_spatial', action='store_true', default=False, help='Whether scramble spatial data')
    parser.add_argument('--mixup_train', action='store_true', default=False, help='Whether or mixup train/dev data')
    parser.add_argument('--mixup_by_class', action='store_true', default=False, help='Whether or mixup train data only within a label')
    parser.add_argument('--mixup_k', type=int, default=2, help='Num images to mixup for train')
    parser.add_argument('--mixup_inflate_n', type=float, default=5.0, help='Num images to sample from orig train/dev for mixup')
    parser.add_argument('--scramble_k', type=int, default=2, help='Num coordinates for ')
    parser.add_argument('--scramble_input_dim', type=int, default=2, help='dim of input before scramble. For images, [W*H*C]')

    parser.add_argument('--dataset_rationale_mask_path', default=None, help='If none, flag is ignored. If not none, then preprocess samples by applying rationale mask')
    parser.add_argument('--base_rationale_index', type=int, default=-1, help='Index of rationale to use for first stage.')
    parser.add_argument('--image_augmentations', nargs='*', default=['scale_2d'], help='List of image-transformations to use [default: ["scale_2d"]] \
                        Usage: "--image_augmentations trans1/arg1=5/arg2=2 trans2 trans3/arg4=val"')
    parser.add_argument('--tensor_augmentations', nargs='*', default=['normalize_2d'], help='List of tensor-transformations to use [default: ["normalize_2d"]]\
                        Usage: similar to image_augmentations')
    parser.add_argument('--test_image_augmentations', nargs='*', default=['scale_2d'], help='List of image-transformations to use for the dev and test dataset [default: ["scale_2d"]] \
                        Usage: similar to image_augmentations')
    parser.add_argument('--test_tensor_augmentations', nargs='*', default=['normalize_2d'], help='List of tensor-transformations to use for the dev and test dataset [default: ["normalize_2d"]]\
                        Usage: similar to image_augmentations')
    parser.add_argument('--fix_seed_for_multi_image_augmentations', action='store_true', default=False, help='Whether to use the same seed (same random augmentations) for multi image inputs.')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for each data loader [default: 4]')
    parser.add_argument('--img_size',  type=int, nargs='+', default=[256, 256], help='width and height of image in pixels. [default: [256,256]')
    parser.add_argument('--get_dataset_stats', action='store_true', default=False, help='Whether to compute the mean and std of the training images on the fly rather than using precomputed values')
    parser.add_argument('--img_mean', type=float, nargs='+', default=[0.2023], help='mean value of img pixels. Per channel. ')

    parser.add_argument('--img_std', type=float, nargs='+', default=[0.2576], help='std of img pixels. Per channel. ')
    parser.add_argument('--img_dir', type=str, default='/home/administrator/Mounts/Isilon/pngs16', help='dir of images. Note, image path in dataset jsons should stem from here')
    parser.add_argument('--num_chan', type=int, default=3, help='Number of channels in img. [default:3]')
    parser.add_argument('--force_input_dim', action='store_true', default=False, help='trunctate hiddens from file if not == to input_dim')
    parser.add_argument('--input_dim', type=int, default=512, help='Input dim for 2stage models. [default:512]')
    parser.add_argument('--reduce_dim_multihead', action='store_true', default=False, help='Whether to reduce the dimension of each head in multihead attention pooling')
    parser.add_argument('--use_coarse_for_value', action='store_true', default=False, help='Whether to use the concatenated convs as the value vector in coarse-grained attention pooling')
    parser.add_argument('--multi_image', action='store_true', default=False, help='Whether image will contain multiple slices. Slices could indicate different times, depths, or views')
    parser.add_argument('--input_loader_name', type=str, default=None, help = "Name of loader to use (images, hiddens, etc)")
    parser.add_argument('--load_img_as_npz', action='store_true', default=False, help='All channels of images are stored as one npz')
    parser.add_argument('--load_IRS_as_npy', action='store_true', default=False, help='Load thermal video IRS as NPY')
    parser.add_argument('--use_random_offset', action='store_true', default=False, help='Load thermal video IRS as NPY')
    parser.add_argument('--concat_img_channels', action='store_true', default=False, help='Whether combine images across channels')
    parser.add_argument('--num_images', type=int, default=1, help='In multi image setting, the number of images per single sample.')
    parser.add_argument('--min_num_images', type=int, default=0, help='In multi image setting, the min number of images per single sample.')
    parser.add_argument('--inflation_factor', type=int, default=1, help='In multi image setting, dim of depth to inflate the model to.')
    parser.add_argument('--inflate_time_like_hw', action='store_true', default=False, help='Inflate time depths and strides like 2d')

    # handling CT slices
    parser.add_argument('--padding_method',  type=str, default='evenly', help='How to pad image series with black image. Default is evenly distrubted across slices to obtain num_images slices.')
    parser.add_argument('--truncation_method',  type=str, default='evenly', help='How to select slices if image series has more slices than num_images.')
    parser.add_argument('--slice_thickness_filter',  type=float, nargs='*', help='Slice thickness using, if restricting to specific thickness value.')
    parser.add_argument('--cross_section_filter',  type=str, nargs='*', help='Restrict to using specific cross sections [transverse, coronal, sagittal, oblique].')
    parser.add_argument('--use_only_thin_cuts_for_ct', action='store_true', default=False, help='Wether to use image series with thinnest cuts only.')
    
    # multi image pooling
    parser.add_argument('--multipool_pools', type=str, nargs = '*', help='List of pools and relevant args to use in succession or aggregate into one pooled representation. Usage: "--multipool_pools/arg1=val1/arg2=val2" ')
    parser.add_argument('--image_pool_name', type=str, default='Simple_AttentionPool', help='pool to perform over a single image (slice).')
    parser.add_argument('--volume_pool_name', type=str, default='GlobalMaxPool', help='pool to perform over a volume of images, aggregating multiple slices.')
    parser.add_argument('--conv_pool_kernel_size', type=int, default= 5, help='kernel size of convolution over slices.')

    # region annotations
    parser.add_argument('--load_annotations_from_hidden', action='store_true', default=False, help='Load precomputed annotations.') ## rm
    parser.add_argument('--use_region_annotations', action = 'store_true', default = False, help = 'whether to use image-level annotations (pixel labels) in modeling')
    parser.add_argument('--use_volume_annotations', action = 'store_true', default = False, help = 'whether to use volume-level annotations (image labels) in modeling')
    parser.add_argument('--region_annotations_filepath', type = str , help = 'Path to annotations file')
    parser.add_argument('--predict_volume_attentions', action = 'store_true', default = False, help = 'Whether to predict attention scores over volume using annotations. Guided attention loss.')
    parser.add_argument('--predict_image_attentions', action = 'store_true', default = False, help = 'Whether to predict attention scores over single image using annotations. Guided attention loss.')
    parser.add_argument('--annotation_loss_lambda', type = float, default = 1, help = 'Weight of annotation losses')
    parser.add_argument('--image_attention_loss_lambda', type = float, default = 1, help = 'Weight of loss for predicting image attention scores')
    parser.add_argument('--volume_attention_loss_lambda', type = float, default = 1, help = 'Weight of loss for predicting volume attention scores')
    parser.add_argument('--save_attention_scores', action = 'store_true', default = False, help = 'Whether to save attention scores when using attention mechanism')

    parser.add_argument('--metadata_dir', type=str, default='/home/administrator/Mounts/Isilon/metadata', help='dir of metadata jsons.')
    parser.add_argument('--cache_path', type=str, default=None, help='dir to cache images.')
    parser.add_argument('--cache_full_img', action='store_true', default=False, help='Cache full image locally as well as cachable transforms')


    # sampling
    parser.add_argument('--class_bal', action='store_true', default=False, help='Wether to apply a weighted sampler to balance between the classes on each batch.')
    parser.add_argument('--split_probs', type=float, nargs='+', default=[0.6, 0.2, 0.2], help='Split probs for datasets without fixed train dev test. ')
    # rationale
    parser.add_argument('--nlp_model_name', type=str, default='mlp', help="Form of model as encoder for nlp model.")
    parser.add_argument('--rationale_sparsity_lambda', type=float, default=0,  help='lambda to weigh the ratioanle sparisity loss. encourage rationale selection to be sparse')
    parser.add_argument('--rationale_coherence_lambda', type=float, default=0,  help='lambda to weigh the rationale coherece loss. Encourage rationale to pick words next to each other')
    parser.add_argument('--rationale_complement_lambda', type=float, default=0,  help='lambda to weigh the complemenet lambda loss. Encourage rationale complement to not be informative.')
    parser.add_argument('--rationale_synergy_lambda', type=float, default=0,  help='lambda to weigh the ratioanle residual sup loss. encourage each rationale to be predictive above and beyond')
    parser.add_argument('--rationale_independence_lambda', type=float, default=0,  help='lambda to weigh the rationale independence loss. encourage each rationale to be statistically independent')
    parser.add_argument('--rationale_distribution_lambda', type=float, default=0,  help='lambda to weigh the ratioanle distribution loss. encourage rationale selection to be sparse')
    parser.add_argument('--num_rationale', type=int, default=1,  help='Num rationales to use.')
    parser.add_argument('--baseline_knowledge_dim', type=int, default=0, help='Dim of baseline knowledge tensor.')
    parser.add_argument('--disable_select_as_irrelevant', action='store_true',  help='Set to true disable selecting a token as irrelevant for rationale models.')
    parser.add_argument('--gumble_tau', type=float, default=1.0,  help='Gumble softmax tau')
    parser.add_argument('--gumble_tau_decay', type=float, default=0.9999,  help='Gumble softmax tau decay')
    parser.add_argument('--gumble_tau_min', type=float, default=0.01,  help='Gumble softmax tau min')

    parser.add_argument('--rationale_residual_grad_rev_at_generator', action='store_true',  help='Set to true  to set only generator to maximize synergy. Set to false for both generator and encoder to max synergy.')

    # regularization
    parser.add_argument('--use_adv', action='store_true', default=False, help='Wether to add a adversarial loss representing the kl divergernce from source to target domain. Note, dataset obj must provide "target_x" to take effect.')
    parser.add_argument('--adv_loss_lambda',  type=float, default=0.5,  help='lambda to weigh the adversarial loss.')
    parser.add_argument('--primary_loss_lambda', type=float, default=1.0,  help='lambda to weigh the primary loss.')
    parser.add_argument('--adv_on_logits_alone',  action='store_true', default=False, help='Train adversary using only posterior dist.')
    parser.add_argument('--num_model_steps',  type=int, default=1,  help='num steps of model optimization before switching to adv optimization.')
    parser.add_argument('--num_adv_steps',  type=int, default=100,  help='max num steps of adv before switch back to model optimization. ')
    # storing hiddens
    parser.add_argument('--store_hiddens', action='store_true', default=False, help='Save hidden repr from each image to an npz based off results path, git hash and exam name')
    parser.add_argument('--save_predictions', action='store_true', default=False, help='Save hidden repr from each image to an npz based off results path, git hash and exam name')
    parser.add_argument('--hiddens_dir', type=str, default='hiddens/test_run', help='Dir to store hiddens npy"s when store_hiddens is true')
    # risk factors
    parser.add_argument('--use_risk_factors', action='store_true', default=False, help='Whether to feed risk factors into last FC of model.') #
    parser.add_argument('--pred_risk_factors', action='store_true', default=False, help='Whether to predict value of all RF from image.') #
    parser.add_argument('--pred_risk_factors_lambda',  type=float, default=0.25,  help='lambda to weigh the risk factor prediction.')
    parser.add_argument('--use_pred_risk_factors_at_test', action='store_true', default=False, help='Whether to use predicted risk factor values at test time.') #
    parser.add_argument('--use_pred_risk_factors_if_unk', action='store_true', default=False, help='Whether to use predicted risk factor values at test time only if rf is unk.') #
    parser.add_argument('--risk_factor_keys', nargs='*', default=[], help='List of risk factors to include in risk factor vector.')
    parser.add_argument('--risk_factor_metadata_path', type=str, default='/home/administrator/Mounts/Isilon/metadata/risk_factors_jul22_2018_mammo_and_mri.json', help='Path to risk factor metadata file.')
    parser.add_argument('--drug_activity_json_path', type=str, default='all_drug_activity_jun19_2020.json', help='Path to risk factor metadata file.')

    #self supervised:
    parser.add_argument('--instance_discrim', action='store_true', default=False, help='Do instance discrim task, i.e byol or moco.') #
    parser.add_argument('--use_pretrained_encoder', action='store_true', default=False, help='Use model pretrained on local task.')

    parser.add_argument('--contrastive_model_name', type=str, default = 'moco', help='Type of contrastive model to apply for instance discrim')
    parser.add_argument('--contrastive_setup', type=str, default = 'simclr', help='Setup name regarding projection heads and losses for models with similar backbones.')

    parser.add_argument('--use_data_augmentation_as_pos', action='store_true', default=False, help='If true, consider pos pairs as data aug of same image. If false, use other image of same drug') #
    parser.add_argument('--projection_dim', type=int, default=128, help='intermediate dim for projection and pred ffns in BYOL / Moco')
    parser.add_argument('--byol_dim', type=int, default=256, help='dim of latent code in BYOL')
    
    parser.add_argument('--encoder_dim', type=int, default=2048, help='dim from last output layer of an encoder.')
    parser.add_argument('--byol', action='store_true', default=False, help='Add bootsrap own latents loss (Byol).') #
    parser.add_argument('--byol_momentum', type=float, default=0.99, help='Momentum to use for updating target network (Byol).') #
    parser.add_argument('--byol_lambda', type=float, default=1.0, help='Lambda to use for BYOL loss.') #
    parser.add_argument('--pred_byol_only', action='store_true', default=False, help='multiply all other losses by 0 (Byol).') #

    parser.add_argument('--moco', action='store_true', default=False, help='Use Moco.') #
    parser.add_argument('--moco_queue_size', type=int, default=65536, help='Length of queue.') #
    parser.add_argument('--moco_momentum', type=float, default=0.999, help='Momentum for key encoder.') #
    parser.add_argument('--moco_temprature', type=float, default=0.07, help='Temprature for noise contrastive loss.') #

    #survival analysis setup
    parser.add_argument('--survival_analysis_setup', action='store_true', default=False, help='Whether to modify model, eval and training for survival analysis.') #
    parser.add_argument('--make_probs_indep', action='store_true', default=False, help='Make surival model produce indepedent probablities.') #
    parser.add_argument('--mask_mechanism', default='default', help='How to mask for survival objective. options [default, indep, slice, linear].') #
    parser.add_argument('--eval_survival_on_risk', action='store_true', default=False, help='Port over survival model to risk model.') #
    parser.add_argument('--max_followup', type=int, default=5, help='Max followup to predict over')
    parser.add_argument('--eval_risk_on_survival', action='store_true', default=False, help='Port over risk model to survival model.') #
    parser.add_argument('--survival_formulation', type=str, default='mirai', help='Mode of Cumulative_Probability_Layer.') #
    
    # progression-free survival
    parser.add_argument('--treatment_response_task', type=str, default = 'mskcc_binary_bor', help = 'Name of task for MSK response to treatment label.')
    parser.add_argument('--progression_free_months', type=float, default = 12, help = 'Number of months of progression free survival to be considered as a responder')

    #recursion task
    parser.add_argument('--pred_drug_activ', action='store_true', default=False, help='Whether to use predict drug activity as dev evalution')
    parser.add_argument('--drug_active_lambda', type=float, default=1.0,  help='lambda to weigh the drug_activ loss.')
    parser.add_argument('--use_linearizing_pool', action='store_true', default=False, help='Whether to use linearizing function to aggregate representations.') #
    parser.add_argument('--use_log_conc', action='store_true', default=False, help='Whether to use log conc in  linearizing function to aggregate representations.') #
    parser.add_argument('--sample_log_half', action='store_true', default=False, help='Make second image half log conc of first image.') #
    parser.add_argument('--linearizing_lambda', type=float, default=50.0,  help='lambda to weigh the linearity loss.')
    parser.add_argument('--pred_drug_activ_aggr_mode',type=str, default='max', help='How to get a single drug represnetation from list of image represnetations')
    parser.add_argument('--use_only_max_conc',  action='store_true', default=False, help='Whether to exclude all images that arent max conc for that drug')
    parser.add_argument('--sample_all_max',  action='store_true', default=False, help='Whether to exclude all images that arent max conc for that drug for multi conc')
    parser.add_argument('--use_only_known_activ',  action='store_true', default=False, help='Whether to exclude all images for which activ is known')
    parser.add_argument('--exclude_mock',  action='store_true', default=False, help='Exclude mock from training')
    parser.add_argument('--exclude_sars',  action='store_true', default=False, help='Exclude sars from training')
    parser.add_argument('--always_sample_sars',  action='store_true', default=False, help='Always sample sars in multi conc reid training')

    # chemprop
    parser.add_argument('--graph_dummy',  action='store_true', default=False, help='Dummy trainer')
    parser.add_argument('--use_chemprop',  action='store_true', default=False, help='Initialize GCN')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='Use messages on atoms instead of messages on bonds')
    parser.add_argument('--hidden_size', type=int, default=300,
                    help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network') # Force False
    parser.add_argument('--use_input_features', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network') # Force False
    parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU'],
                        help='Activation function')
    parser.add_argument('--mask_all_bonds', action='store_true', default=False, help="Whether to mask all bonds in MPN")

    # nlp
    parser.add_argument('--vocab_file',  type=str, default=None, help='Path to vocabulary file')

    # hiddens based dataset
    parser.add_argument('--use_precomputed_hiddens', action='store_true', default=False, help='Whether to only use hiddens from a pretrained model. By default, assume all hiddens stored in one large pickle produced by scripts/get_hiddens.py')
    parser.add_argument('--use_precomputed_hiddens_in_get_hiddens', action='store_true', default=False, help='Whether to only use hiddens from a pretrained model.')
    parser.add_argument('--hiddens_results_path', type=str, default='/home/administrator/Mounts/Isilon/results/hiddens_from_best_dev_aug_29.results.json', help='Path to results file with hiddens for the whole dataset.')
    parser.add_argument('--hiddens_stored_as_individual_numpy_arrays', action='store_true', default=False, help='Assume hiddens were stored using "store_hiddens" flag and were stored using individual paths.')
    parser.add_argument('--collate_multi_image_hidden_paths', action='store_true', default=False, help='Whether to join paths into one string to load hiddens.')

    # learning
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer to use [default: adam]')
    parser.add_argument('--objective', type=str, default="cross_entropy", help='objective function to use [default: cross_entropy]')
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum to use with SGD')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='initial learning rate [default: 0.5]')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 Regularization penaty [default: 0]')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs without improvement on dev before halving learning rate and reloading best model [default: 5]')

    parser.add_argument('--tuning_metric', type=str, default='loss', help='Metric to judge dev set results. Possible options include auc, loss, accuracy [default: loss]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 128]')
    parser.add_argument('--dropout', type=float, default=0.25, help='Amount of dropout to apply on last hidden layer [default: 0.25]')
    parser.add_argument('--save_dir', type=str, default='snapshot', help='where to dump the model')
    parser.add_argument('--results_path', type=str, default='logs/test.args', help='where to save the result logs')
    parser.add_argument('--project_name', type=str, default='sandstone-sandbox', help='Name of project for comet logger')
    parser.add_argument('--workspace', type=str, default='yala', help='Name of workspace for comet logger')
    parser.add_argument('--comet_tags', nargs='*', default=[], help="List of tags for comet logger")
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of data to use, i.e 1.0 for all and 0 for none. Used for learning curve analysis.')
    parser.add_argument('--dataset_path', type=str, default='/dev/null', help='path to dataset (for k-fold cross validation experiments)')

    # Alternative training/testing schemes
    parser.add_argument('--cross_val_seed', type=int, default=0, help="Seed used to generate the partition.")
    parser.add_argument('--assign_splits', action='store_true', default=False, help = "Whether to assign different splits than those predetermined in dataset")
    parser.add_argument('--split_type', type=str, default='random', help="How to split dataset if assign_split = True. Usage: ['random', 'institution_split'].")

    parser.add_argument('--model_name', type=str, default='resnet18', help="Form of model, i.e resnet18, aggregator, revnet, etc.")
    parser.add_argument('--trainer_name', type=str, default='default', help="Form of model, i.e resnet18, aggregator, revnet, etc.")
    parser.add_argument('--num_layers', type=int, default=3, help="Num layers for transformer based models.")
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--state_dict_path', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--img_encoder_model', type=str, default=None, help='model name to load for transformer encoder. Only used for mirai_full type models [default: None]')
    parser.add_argument('--img_encoder_snapshot', type=str, default=None, help='filename of img_feat_extractor model snapshot to load. Only used for mirai_full type models [default: None]')
    parser.add_argument('--transformer_snapshot', type=str, default=None, help='filename of transformer model snapshot to load. Only used for mirai_full type models [default: None]')
    parser.add_argument('--pretrained_on_imagenet', action='store_true', default=False, help='Pretrain the model on imagenet. Only relevant for default models like VGG, resnet etc')
    parser.add_argument('--pretrained_imagenet_model_name', type=str, default='resnet18', help='Name of pretrained model to load for custom resnets.')
    
    parser.add_argument('--pretrained_transformer_model_name', type=str, default='distilbert-base-uncased', help='Name of pretrained model to load for transformer.')
    parser.add_argument('--fix_pretrained_transformer', action='store_true', default = False, help='Whether to use transformer for embedding or train.')

    # MLP
    parser.add_argument('--mlp_pool_name', type=str, default='GlobalMaxPool', help='Pool to perform over the representations of a volume of images inside an MLP.')
    parser.add_argument('--mlp_layer_configuration', type=int, nargs='*', help='Defines output dimensions for hidden layers of the multi_image_mlp model when a list of ints is given. [default: None]')
    parser.add_argument('--mlp_predict_classes',  action='store_true', default=False, help= 'Whether to use MLP to predict class or just as feed-forward module')
    parser.add_argument('--mlp_regression',  action='store_true', default=False, help= 'Whether to use MLP for regression')
    parser.add_argument('--mlp_lr', type = float, default = 3e-4, help= 'Arg to use when need to set different lr for mlp than base model')

    # transformer
    parser.add_argument('--hidden_dim', type=int, default=512, help='start hidden dim for transformer')
    parser.add_argument('--num_heads', type=int, default=8, help='Num heads for transformer')
    parser.add_argument('--positional_encoding', type=str, help='Type of positional encoding. One of [relative, absolute]')
    parser.add_argument('--transfomer_hidden_dim', type=int, default=512, help='start hidden dim for transformer')
    parser.add_argument('--transformer_pool_name', type=str, default='GlobalAvgPool', help='Pooling mechanism for transformer')
    parser.add_argument('--use_skip_connect_in_transformer', action='store_true', default=False, help='Use a skip/residual connection across transformer')
    parser.add_argument('--use_skip_connect_in_transformer_layerwise', action='store_true', default=False, help='Use a skip/residual connection across transformer')
    parser.add_argument('--hidden_transformer_embeddings', type=str, nargs='*', help='List of embeddings to be done. Usage for a single embedding: "--hidden_transformer_embeddings embedding_name/size=val/dim=value"')

    # resnet-specific
    parser.add_argument('--block_layout', type=str, nargs='+', default=["BasicBlock,2", "BasicBlock,2", "BasicBlock,2", "BasicBlock,2"], help='Layout of blocks for a ResNet model. Must be a list of length 4. Each of the 4 elements is a string of form "block_name,num_repeats-block_name,num_repeats-...". [default: resnet18 layout]')
    parser.add_argument('--block_widening_factor', type=int, default=1, help='Factor by which to widen blocks.')
    parser.add_argument('--num_groups', type=int, default=1, help='Num groups per conv in Resnet blocks.')
    parser.add_argument('--gn_num_groups', type=int, default=1, help='Num groups in group normalization. 1 is equivalent to layernorm')
    parser.add_argument('--replace_bn_with_gn', action='store_true', default=False, help='Replace all instances of BatchNorm with GroupNorm')
    parser.add_argument('--pool_name', type=str, default='GlobalAvgPool', help='Pooling mechanism')

    # visualize_attributions_utils
    parser.add_argument('--overlay', action = 'store_true', default=False, help="Overlay attributions on original image")
    parser.add_argument('--outlines', action = 'store_true', default=False, help="Visualize outlines of int gradients")
    parser.add_argument('--polarity', type = str, default='positive', help="Show pos, neg, or both types of int gradients")
    parser.add_argument('--clip_above_percentile', type = float, default= 99.0, help="Percentile above which attributions are ingored")
    parser.add_argument('--clip_below_percentile', type = float, default=0, help="Percentile below which attributions are ingored")
    parser.add_argument('--morphological_cleanup', action = 'store_true', default=False, help="Show a coarser visualization of attributions")
    parser.add_argument('--outlines_component_percentage', type = float, default=90, help="Which attributions to keep for outline")
    parser.add_argument('--plot_distribution', action = 'store_true', default=False, help="Whether to plot distribution of attributions")
    parser.add_argument('--mask_mode', action = 'store_true', default=False, help="Visualization style")
    # inference
    parser.add_argument('--inference_mode', action='store_true', default=False, help="Use flag to tell model to output logits only")
    parser.add_argument('--inference_method_name', type = str, default = 'saliency', help="Name of inference method")
    parser.add_argument('--additional_inference_parameters', type = str, nargs = '*', help="Additional parameters accepted by inference method")
    parser.add_argument('--downsample_attributions', action='store_true', default=False, help="Whether to downsample input (image) attributions. Downsample using AvgPool with stride of 2.")
    parser.add_argument('--store_full_attributions', action='store_true', default=False, help="Whether to store attribution maps. Stored as np arrays in hiddens_dir")
    parser.add_argument('--inference_postprocessing_methods', type = str, nargs = '*', help="Computations to perform on attribution maps")
    parser.add_argument('--snapshot_results_path', type = str, help="Path to results dict produced from same experiment as the snapshot's model")

    parser.add_argument('--run_notes', type = str, help="Unused argument. Useful in configs to simulate multiple runs of same parameters")
    
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

    if args.debug:
        args.num_workers = 0
        args.limit_train_batches = 10
        args.limit_val_batches = 10
        args.limit_test_batches = 0.1
        args.comet_tags = 'debug'

    if 'rationale' in args.model_name:
        args.lightning_name = 'rationale'

    if args.mixup_train or args.scramble_spatial:
        args.lightning_name = 'private'
        args.model_name = 'hiddens_mlp'

    # Set args particular to dataset
    get_dataset_class(args).set_args(args)

    args.unix_username = pwd.getpwuid( os.getuid() )[0]

    # using annotations
    args.use_annotations = args.use_volume_annotations or args.use_region_annotations
    
    # pretrained model
    if args.use_pretrained_encoder:
        assert args.snapshot is not None, 'ARGS ERROR! --use_pretrained_encoder flag used while --snapshot is None'

    # learning initial state
    args.step_indx = 1

    # Parse list args to appropriate data format
    parse_list_args(args)

    # Check whether certain args or arg combinations are valid
    validate_args(args)

    return args

def validate_args(args):
    """Checks whether certain args or arg combinations are valid.

    Raises:
        Exception if an arg or arg combination is not valid.
    """


    if args.survival_analysis_setup != ('_full_future' in args.dataset):
        raise ValueError(INVALID_DATASET_FOR_SURVIVAL)

    if (args.byol) and not (args.instance_discrim and args.multi_image and args.num_images == 2):

        raise ValueError(SELF_SUPER_ERROR)

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

    args.block_layout = parse_block_layout(args.block_layout)
    args.hidden_transformer_embeddings = parse_embeddings(args.hidden_transformer_embeddings) if args.hidden_transformer_embeddings is not None else None
