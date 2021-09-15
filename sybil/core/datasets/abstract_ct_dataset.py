from sandstone.datasets.abstract_dataset import Abstract_Dataset
import sandstone.utils
import tqdm
import numpy as np
from collections import Counter, defaultdict
import os 
from sandstone.utils.region_annotation import get_scaled_annotation_mask
import warnings
import traceback
import torch 

IMG_PAD_PATH = '/Mounts/rbg-storage1/datasets/MSKCC/other/pad.tif'
DCM_PAD_PATH = '/Mounts/rbg-storage1/datasets/NLST/dicom_pad_token.dcm'
ALIGNMENT_NUM_IMGS = 200
CT_ITEM_KEYS = ['cancer_location', 'num_original_slices', 'cancer_laterality', 'has_annotation','use_manual_annotation', 'volume_annotations','annotation_areas']

class Abstract_CT_Dataset(Abstract_Dataset):
    def __getitem__(self, index):
        sample = self.dataset[index]
        try:
            item = super().__getitem__(index)

            if self.args.use_region_annotations:
                mask = item.pop('mask')
                if not self.args.load_annotations_from_hidden:
                    mask = torch.abs(mask)
                    mask_area = mask.sum(dim=(-1,-2) ).unsqueeze(-1).unsqueeze(-1)
                    mask_area[mask_area==0] = 1
                    mask = mask/mask_area
                    item['image_annotations'] = mask
                else:
                    item['image_annotations']= sample['image_annotations']
                item['slice_ids'] = sample['slice_ids']
            
            for key in CT_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]
            
            return item
        except Exception:
            warnings.warn('CT DATASET: Failed to load.\nException {}:'.format(traceback.print_exc())) 
    
    @property
    def PAD_PATH(self):
        if self.args.input_loader_name == 'dicom_loader':
            return DCM_PAD_PATH
        else:
            return IMG_PAD_PATH

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
    
    def fit_to_length(self, arr, pad_token, max_length, truncate_method = 'evenly', pad_method = 'evenly', start_index = 0):
        '''
        Fits arr to max_length by either truncating arr (remove excess from both tails) 
        or padding (on both sides) arr with pad_token.
        '''
        if truncate_method == 'none' and pad_method == 'none':
            return arr 
            
        if len(arr) > max_length:
            arr = truncate(arr, max_length, truncate_method, start_index)

        elif len(arr) < max_length:
            arr = pad(arr, pad_token, max_length, pad_method)
        return arr

    def prep_contrastive_setup(self, sample, paths, locations, examid, seriesid, args):
        sample_idx = np.random.choice(len(paths), size = 1).item()
        sample['paths'] = [paths[sample_idx]]*2
        sample['slice_locations'] = [locations[sample_idx]]*2
        sample['y'] = 0
        sample['exam'] =  '{}_{}_{}'.format(examid, seriesid, locations[sample_idx])
        
        return sample 


def truncate(arr, max_length, method, start_index = 0):
    if method == 'two_tailed':
        start_idx = (len(arr) - max_length + 1) // 2
        arr = arr[start_idx : start_idx + max_length]
    if method == 'evenly':
        include_ids = np.round(np.linspace(0, len(arr) - 1, max_length)).astype(int)
        arr = [elt for idx, elt in enumerate(arr) if idx in include_ids]
    if method == 'from_start_index':
        arr = arr[start_index: (start_index+max_length)]
    return arr

def pad(arr, pad_token, max_length, method):
    num_pad_tokens = max_length - len(arr)
    if method == 'two_tailed':
        arr = [pad_token] * ((num_pad_tokens + 1) // 2) + arr + [pad_token] * ((num_pad_tokens) // 2)
    if method =='right_tail':
        arr = arr + [pad_token]*(num_pad_tokens)
    if method == 'left_tail':
        arr = [pad_token]*(num_pad_tokens) + arr
    if method == 'evenly':
        pad_ids = np.round(np.linspace(0, max_length - 1, num_pad_tokens)).astype(int)
        for idx in pad_ids:
            arr.insert(idx, pad_token)

    return arr
    
