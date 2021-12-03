import numpy as np 
import math 

def order_slices( img_paths, slice_locations):
    sorted_ids = np.argsort(slice_locations)
    sorted_img_paths = np.array(img_paths)[sorted_ids].tolist()
    sorted_slice_locs = np.sort(slice_locations).tolist()
    return sorted_img_paths, sorted_slice_locs


def fit_to_length(arr, pad_token, max_length, truncate_method = 'evenly', pad_method = 'evenly', start_index = 0):
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
    
def get_scaled_annotation_mask(additional, args, scale_annotation=True):
    '''
    Construct bounding box masks for annotations
    Args:
        - additional['image_annotations']: list of dicts { 'x', 'y', 'width', 'height' }, where bounding box coordinates are scaled [0,1]. 
        - args
    Returns:
        - mask of same size as input image, filled in where bounding box was drawn. If additional['image_annotations'] = None, return empty mask. Values correspond to how much of a pixel lies inside the bounding box, as a fraction of the bounding box's area
    '''
    H, W = args.img_size
    mask = np.zeros((H, W)) 
    if additional['image_annotations'] is None:
        return mask
    
    for annotation in additional['image_annotations']:
        single_mask = np.zeros((H, W)) 
        x_left, y_top = annotation['x'] * W, annotation['y'] * H
        x_right, y_bottom = x_left + annotation['width'] * W, y_top + annotation['height'] * H

        # pixels completely inside bounding box
        x_quant_left, y_quant_top = math.ceil(x_left), math.ceil(y_top)
        x_quant_right, y_quant_bottom = math.floor(x_right), math.floor(y_bottom)

        # excess area along edges
        dx_left = x_quant_left - x_left
        dx_right = x_right - x_quant_right
        dy_top = y_quant_top - y_top
        dy_bottom = y_bottom - y_quant_bottom 

        # fill in corners first in case they are over-written later by greater true intersection
        # corners
        single_mask[ math.floor(y_top), math.floor(x_left)] = dx_left * dy_top
        single_mask[ math.floor(y_top), x_quant_right ] = dx_right * dy_top
        single_mask[ y_quant_bottom, math.floor(x_left)] = dx_left * dy_bottom
        single_mask[ y_quant_bottom, x_quant_right] = dx_right * dy_bottom

        # edges 
        single_mask[ y_quant_top: y_quant_bottom, math.floor(x_left) ] = dx_left
        single_mask[ y_quant_top: y_quant_bottom, x_quant_right ] = dx_right
        single_mask[ math.floor(y_top), x_quant_left: x_quant_right ] = dy_top
        single_mask[ y_quant_bottom , x_quant_left: x_quant_right ] = dy_bottom

        # completely inside
        single_mask[ y_quant_top: y_quant_bottom , x_quant_left: x_quant_right] = 1 
        
        # in case there are multiple boxes, add masks and divide by total later 
        mask += single_mask

    if scale_annotation:
        mask /= mask.sum()
    return mask
