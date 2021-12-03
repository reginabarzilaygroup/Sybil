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
from sandstone.utils.generic import log
import pdb
import copy
import torch 

# Error Messages
METAFILE_NOTFOUND_ERR = "Metadata file {} could not be parsed! Exception: {}!"
LOAD_FAIL_MSG = "Failed to load image: {}\nException: {}"

# Constants
IMG_PAD_PATH = 'datafiles/pad.tif'
ALIGNMENT_NUM_IMGS = 200
DEBUG_SIZE=1000


DATASET_ITEM_KEYS = ['pid', 'exam', 'series', 'y_seq', 'y_mask', 'time_at_event', 'cancer_laterality', 'has_annotation',  'volume_annotations','annotation_areas']


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
        args.metadata_path = os.path.join(args.metadata_file)

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
            
        self.dataset = self.create_dataset(split_group, args.img_dir)
        if len(self.dataset) == 0:
            return
        
        log(self.get_summary_statement(self.dataset, split_group), args)
        
        dist_key = 'y'
        label_dist = [d[dist_key] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1./ len(label_counts)
        label_weights = {
            label: weight_per_label/count for label, count in label_counts.items()
            }
        
        log("Class counts are: {}".format(label_counts), args)
        log("Label weights are {}".format(label_weights), args)
        self.weights = [ label_weights[d[dist_key]] for d in self.dataset]

    @property
    def PAD_PATH(self):
        return IMG_PAD_PATH

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
            x, mask = self.input_loader.get_images(sample['paths'], sample['additionals'], sample)
            
            item = {
                'x': x,
                'y': sample['y']
                }

            if self.args.use_region_annotations:
                if not self.args.load_annotations_from_hidden:
                    mask = torch.abs(mask)
                    mask_area = mask.sum(dim=(-1,-2) ).unsqueeze(-1).unsqueeze(-1)
                    mask_area[mask_area==0] = 1
                    mask = mask/mask_area
                    item['image_annotations'] = mask
                else:
                    item['image_annotations']= sample['image_annotations']
                item['slice_ids'] = sample['slice_ids']
            
            for key in DATASET_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample['paths'], traceback.print_exc()))  
    
    def get_ct_annotations(self, sample):
        # correct empty lists of annotations
        if sample['series'] in self.annotations_metadata:
            self.annotations_metadata[ sample['series'] ] = {k: v for k, v in self.annotations_metadata[ sample['series'] ].items() if len(v) > 0 }

        if sample['series'] in self.annotations_metadata:
            # check if there is an annotation in a slice
            sample['volume_annotations'] = np.array([ int( os.path.splitext(os.path.basename(path))[0]  in self.annotations_metadata[ sample['series'] ] ) for path in sample['paths'] ] )
            # store annotation(s) data (x,y,width,height) for each slice
            sample['additionals'] = [ {'image_annotations': self.annotations_metadata[ sample['series'] ].get( os.path.splitext(os.path.basename(path))[0], None ) } for path in sample['paths'] ]
        else:
            sample['volume_annotations'] = np.array([ 0  for _ in sample['paths'] ])
            sample['additionals'] = [ {'image_annotations': None }  for path in sample['paths'] ]
        return sample

    def annotation_summary_msg(self, dataset):
        annotations = [np.sum(d['volume_annotations']) for d in dataset ]
        annotation_dist =  Counter(annotations)
        annotation_dist = dict( sorted( annotation_dist.items(), key = lambda i: i[0] ) )
        num_annotations = sum([ i > 0 for i in annotations])
        mean_dist = np.mean([k for k in annotation_dist.values() if k!=0])
        return '\nAnnotations: Dataset has {} annotated samples. Number of annotations per sample has the following distribution {}, with mean {} \n'.format(num_annotations, annotation_dist, mean_dist)
                                                        

    def get_scaled_annotation_area(self, sample):
        '''
        no_box = [{'width': 0, 'height': 0}]
        if sample['series'] in self.annotations_metadata:
            # total area of bounding boxes in 
            areas_per_slice = [ [ box['width']*box['height'] for box in self.annotations_metadata[ sample['series'] ].get( os.path.splitext(os.path.basename(path))[0], no_box ) ] for path in sample['paths'] ]
            return np.array( [ np.sum(areas) for areas in areas_per_slice] )
        else:
            return np.array([ 0  for _ in sample['paths'] ])
        '''
        areas = []
        for additional in sample['additionals'] :
            mask = get_scaled_annotation_mask(additional, self.args, scale_annotation=False)
            areas.append( mask.sum()/ ( mask.shape[0] * mask.shape[1] ) )
        return np.array(areas)
    

    

