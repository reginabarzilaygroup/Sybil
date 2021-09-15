import numpy as np
import pickle
from abc import ABCMeta, abstractmethod
import torch
from torch.utils import data
import os
import warnings
import json
import traceback
from collections import Counter
from sandstone.datasets.loader.factory import get_input_loader
from sandstone.utils.risk_factors import parse_risk_factors, RiskFactorVectorizer
from scipy.stats import entropy
from sandstone.utils.generic import log
from sandstone.utils.generic import get_path_for_x
import pdb
import copy
import torch 

METAFILE_NOTFOUND_ERR = "Metadata file {} could not be parsed! Exception: {}!"
LOAD_FAIL_MSG = "Failed to load image: {}\nException: {}"

DEVICE_TO_ID = {'Lorad Selenia': 1,
                'Hologic Selenia': 1,
                'Senograph DS ADS_43.10.1':2,
                'Selenia Dimensions': 0,
                'Selenia Dimensions C-View':3}
DEBUG_SIZE=1000


DATASET_ITEM_KEYS = ['ssn', 'pid', 'exam', 'series', 'birads', 'y_seq', 'y_mask', 'time_at_event', 'device', 'device_is_known',
            'time_seq', 'view_seq', 'side_seq', 'y_l', 'y_r', 'y_seq_r', 'y_mask_r', 'time_at_event_r', 'y_seq_l', 'y_mask_l', 
            'time_at_event_l', 'drug_activ', 'drug_activ_known', 'concetrations', 'molecule', 'baseline_knowledge', 'context_seq', 'query_seq', 
            'padded_indices', 'text', 'x_text', 'pd-l1_score', 'tmb', 'omit_sample']


class Abstract_Dataset(data.Dataset):
    """
    Abstract Object for all Datasets. All datasets have some metadata
    property associated with them, a create_dataset method, a task, and a check
    label and get label function.
    """
    __metaclass__ = ABCMeta

    def __init__(self, args, augmentations, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        '''
        super(Abstract_Dataset, self).__init__()
        args.metadata_path = os.path.join(args.metadata_dir,
                                          self.METADATA_FILENAME)

        self.split_group = split_group
        self.args = args
        self.input_loader = get_input_loader(args.cache_path, augmentations, args)
        try:
            self.metadata_json = json.load(open(args.metadata_path, 'r'))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.metadata_path, e))

        if args.debug and isinstance(self.metadata_json,list):
            self.metadata_json = self.metadata_json[:DEBUG_SIZE]
        
        if args.use_annotations or self.args.region_annotations_filepath is not None:
            assert self.args.region_annotations_filepath, 'ANNOTATIONS METADATA FILE NOT SPECIFIED'
            self.annotations_metadata = json.load(open(self.args.region_annotations_filepath, 'r'))
            
        self.path_to_hidden_dict = {}
        self.dataset = self.create_dataset(split_group, args.img_dir)
        if len(self.dataset) == 0:
            return
        if split_group == 'train' and self.args.data_fraction < 1.0:
            self.dataset = np.random.choice(self.dataset, int(len(self.dataset)*self.args.data_fraction), replace=False)
        try:
            self.add_device_to_dataset()
        except:
            log("Could not add device information to dataset", args)
        for d in self.dataset:
            if 'exam' in d and 'year' in d:
                args.exam_to_year_dict[d['exam']] = d['year']
            if 'device_name' in d and 'exam' in d:
                args.exam_to_device_dict[d['exam']] = d['device_name']
        log(self.get_summary_statement(self.dataset, split_group), args)
        args.h_arr, args.w_arr = None, None
        self.risk_factor_vectorizer = None
        if self.args.use_risk_factors and 'nlst' not in self.args.dataset:
            self.risk_factor_vectorizer = RiskFactorVectorizer(args)
            self.add_risk_factors_to_dataset()
        if self.args.pred_drug_activ:
            if 'recursion' in self.args.dataset:
                self.args.drug_to_y = json.load(open(os.path.join(self.args.metadata_dir, self.args.drug_activity_json_path),'r'))
            else:
                assert 'cell_painter' in self.args.dataset
                self.args.drug_to_y = {k:{'active':v['puma_assay_labels'] + v['morpho_assay_labels'] ,'puma_active': v['puma_assay_labels'], 'morpho_active': v['morpho_assay_labels'], 'active_split':v['split']} for k,v in self.metadata_json.items()}

        if 'dist_key' in self.dataset[0]:
            dist_key = 'dist_key'
        else:
            dist_key = 'y'

        label_dist = [d[dist_key] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1./ len(label_counts)
        label_weights = {
            label: weight_per_label/count for label, count in label_counts.items()
            }
        if args.class_bal and args.num_classes < 10:
            log("Class counts are: {}".format(label_counts), args)
            log("Label weights are {}".format(label_weights), args)
        self.weights = [ label_weights[d[dist_key]] for d in self.dataset]

    @property
    @abstractmethod
    def task(self):
        pass

    @property
    @abstractmethod
    def METADATA_FILENAME(self):
        pass

    @abstractmethod
    def check_label(self, row):
        '''
        Return True if the row contains a valid label for the task
        :row: - metadata row
        '''
        pass

    @abstractmethod
    def get_label(self, row):
        '''
        Get task specific label for a given metadata row
        :row: - metadata row with contains label information
        '''
        pass

    def get_summary_statement(self, dataset, split_group):
        '''
        Return summary statement
        '''
        return ""

    @abstractmethod
    def create_dataset(self, split_group, img_dir):
        """
        Creating the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        pass


    @staticmethod
    def set_args(args):
        """Sets any args particular to the dataset."""
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        try:
            if self.args.hiddens_stored_as_individual_numpy_arrays:
                x, mask = self.input_loader.get_image(sample['paths'], sample['additionals'], sample)

            elif self.args.multi_image:
                x, mask = self.input_loader.get_images(sample['paths'], sample['additionals'], sample)

            else:
                if ( ('additional' in sample) and (sample['additional'] is None) ) or ('additional' not in sample):
                    sample['additional'] = {}
                x, mask = self.input_loader.get_image(sample['path'], sample['additional'], sample)

            item = {
                'x': x,
                'y': sample['y']
                }

            for key in DATASET_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            if self.args.use_risk_factors:
                item['risk_factors'] = sample['risk_factors']
            
            if self.args.use_region_annotations:
                item['mask'] = mask
 
            return item
        except Exception:
            path_key =  'paths' if  self.args.multi_image  else 'path'
            warnings.warn(LOAD_FAIL_MSG.format(sample[path_key], traceback.print_exc()))


    def add_risk_factors_to_dataset(self):
        for sample in self.dataset:
            sample['risk_factors'] = self.risk_factor_vectorizer.get_risk_factors_for_sample(sample)

    def add_device_to_dataset(self):
        path_to_device, exam_to_device = self.build_path_to_device_map()
        for d in self.dataset:

            paths = [d['path']] if 'path' in d else d['paths']
            d['device_name'], d['device'], d['device_is_known'] = [], [], []

            for path in paths:
                device = path_to_device[path] if path in path_to_device else 'UNK'
                device_id = DEVICE_TO_ID[device] if device in DEVICE_TO_ID else 0
                device_is_known = device in DEVICE_TO_ID

                d['device_name'].append(device.replace(' ', '_') if device is not None else "<UNK>")
                d['device'].append(device_id)
                d['device_is_known'].append(device_is_known)

            single_image = len(paths) == 1
            if single_image:
                d['device_name'] = d['device_name'][0]
                d['device'] = d['device'][0]
                d['device_is_known'] = d['device_is_known'][0]
            else:
                d['device_name'] = np.array(d['device_name'])
                d['device'] = np.array(d['device'])
                d['device_is_known'] = np.array(d['device_is_known'], dtype=int)

        device_dist = Counter([ d['device'] if single_image else d['device'][-1] for d in self.dataset])
        log("Device Dist: {}".format(device_dist), self.args)
        if self.split_group == 'train':
            device_count = list(device_dist.values())
            self.args.device_entropy = entropy(device_count)
            log("Device Entropy: {}".format(self.args.device_entropy), self.args)

    def build_path_to_device_map(self):
        path_to_device = {}
        exam_to_device = {}
        for mrn_row in json.load(open('/Mounts/phsvna1/CCDS-DRCL/metadata/mammo_metadata_all_years_only_breast_cancer_nov21_2019.json','r')):
            for exam in mrn_row['accessions']:
                exam_id = exam['accession']
                for file, device, view in zip(exam['files'], exam['manufacturer_models'], exam['views']):
                    device_name = '{} {}'.format(device, 'C-View') if 'C-View' in view else device
                    path_to_device[file] = device_name
                    exam_to_device[exam_id] = device_name
        return path_to_device, exam_to_device


    def image_paths_by_views(self, exam):
        '''
        Determine images of left and right CCs and MLO.
        Args:
        exam - a dictionary with views and files sorted relatively.
        returns:
        4 lists of image paths of each view by this order: left_ccs, left_mlos, right_ccs, right_mlos. Force max 1 image per view.
        Note: Validation of cancer side is performed in the query scripts/from_db/cancer.py in OncoQueries
        '''
        source_dir = '/home/{}'.format(self.args.unix_username) if self.args.is_ccds_server else ''

        def get_view(view_name):
            image_paths_w_view = [(view, image_path) for view, image_path in zip(exam['views'], exam['files']) if view.startswith(view_name)]

            if self.args.use_c_view_if_available:
                filt_image_paths_w_view = [(view, image_path) for view, image_path in image_paths_w_view if 'C-View' in view]
                if len(filt_image_paths_w_view) > 0:
                    image_paths_w_view = filt_image_paths_w_view
            else:
                image_paths_w_view = [(view, image_path) for view, image_path in image_paths_w_view if 'C-View' not in view]

            image_paths_w_view = image_paths_w_view[:1]
            image_paths = []
            for _, path in image_paths_w_view:
                image_paths.append(source_dir + path)
            return image_paths


        left_ccs = get_view('L CC')
        left_mlos = get_view('L MLO')
        right_ccs = get_view('R CC')
        right_mlos = get_view('R MLO')
        return left_ccs, left_mlos, right_ccs, right_mlos
