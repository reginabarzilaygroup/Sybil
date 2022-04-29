import numpy as np
import math
from lifelines import KaplanMeierFitter

# Error Messages
METAFILE_NOTFOUND_ERR = "Metadata file {} could not be parsed! Exception: {}!"
LOAD_FAIL_MSG = "Failed to load image: {}\nException: {}"
# Constants
IMG_PAD_TOKEN = "<PAD>"
VOXEL_SPACING = (0.703125, 0.703125, 2.5)
CENSORING_DIST = {
    "0": 0.9851933387778276,
    "1": 0.9748326267838882,
    "2": 0.9659936099043455,
    "3": 0.9587266943913851,
    "4": 0.952360797355704,
    "5": 0.9461857341570268,
}


def order_slices(img_paths, slice_locations):
    sorted_ids = np.argsort(slice_locations)
    sorted_img_paths = np.array(img_paths)[sorted_ids].tolist()
    sorted_slice_locs = np.sort(slice_locations).tolist()
    return sorted_img_paths, sorted_slice_locs


def assign_splits(meta, args):
    for idx in range(len(meta)):
        meta[idx]["split"] = np.random.choice(
            ["train", "dev", "test"], p=args.split_probs
        )


def get_scaled_annotation_mask(additional, args, scale_annotation=True):
    """
    Construct bounding box masks for annotations
    Args:
        - additional['image_annotations']: list of dicts { 'x', 'y', 'width', 'height' }, where bounding box coordinates are scaled [0,1].
        - args
    Returns:
        - mask of same size as input image, filled in where bounding box was drawn. If additional['image_annotations'] = None, return empty mask. Values correspond to how much of a pixel lies inside the bounding box, as a fraction of the bounding box's area
    """
    H, W = args.img_size
    mask = np.zeros((H, W))
    if additional["image_annotations"] is None:
        return mask

    for annotation in additional["image_annotations"]:
        single_mask = np.zeros((H, W))
        x_left, y_top = annotation["x"] * W, annotation["y"] * H
        x_right, y_bottom = (
            x_left + annotation["width"] * W,
            y_top + annotation["height"] * H,
        )

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
        single_mask[math.floor(y_top), math.floor(x_left)] = dx_left * dy_top
        single_mask[math.floor(y_top), x_quant_right] = dx_right * dy_top
        single_mask[y_quant_bottom, math.floor(x_left)] = dx_left * dy_bottom
        single_mask[y_quant_bottom, x_quant_right] = dx_right * dy_bottom

        # edges
        single_mask[y_quant_top:y_quant_bottom, math.floor(x_left)] = dx_left
        single_mask[y_quant_top:y_quant_bottom, x_quant_right] = dx_right
        single_mask[math.floor(y_top), x_quant_left:x_quant_right] = dy_top
        single_mask[y_quant_bottom, x_quant_left:x_quant_right] = dy_bottom

        # completely inside
        single_mask[y_quant_top:y_quant_bottom, x_quant_left:x_quant_right] = 1

        # in case there are multiple boxes, add masks and divide by total later
        mask += single_mask

    if scale_annotation:
        mask /= mask.sum()
    return mask


def get_scaled_annotation_area(sample, args):
    """
    no_box = [{'width': 0, 'height': 0}]
    if sample['series'] in self.annotations_metadata:
        # total area of bounding boxes in
        areas_per_slice = [ [ box['width']*box['height'] for box in self.annotations_metadata[ sample['series'] ].get( os.path.splitext(os.path.basename(path))[0], no_box ) ] for path in sample['paths'] ]
        return np.array( [ np.sum(areas) for areas in areas_per_slice] )
    else:
        return np.array([ 0  for _ in sample['paths'] ])
    """
    areas = []
    for additional in sample["annotations"]:
        mask = get_scaled_annotation_mask(additional, args, scale_annotation=False)
        areas.append(mask.sum() / (mask.shape[0] * mask.shape[1]))
    return np.array(areas)


def get_censoring_dist(train_dataset):
    _dataset = train_dataset.dataset
    times, event_observed = (
        [d["time_at_event"] for d in _dataset],
        [d["y"] for d in _dataset],
    )
    all_observed_times = set(times)
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed)

    censoring_dist = {str(time): kmf.predict(time) for time in all_observed_times}
    return censoring_dist
