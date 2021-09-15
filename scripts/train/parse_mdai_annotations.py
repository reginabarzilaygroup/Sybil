import argparse
import json
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import pdb
import pandas as pd

"""
Script to parse JSON file exported from md.ai
JSON structure is detailed in https://docs.md.ai/data/json/ 

Output is either:
    (1) a dict mapping image (SOPInstanceUID) to list of annotation data
    OR
    (2) dataset object JSON with annotation dict for each image series

Notes:
    - JSON: ['id', 'createdAt', 'updatedAt', 'name', 'description', 'isPrivate', 'users', 'labelGroups', 'datasets']
    
    - meta['labelGroups']: list of dicts with keys: 
        ['id','createdAt', 'updatedAt', 'name', 'description', 'type', 
            'labels': [{'id', 'parentId', 'createdAt', 'updatedAt', 'name', 'shortName', 
            'description', 'color', 'type,', ' scope', 'annotationMode', 'radlexTagIds': []}] ]
    
    - meta['datasets']: list of dicts: ['id', 'type', 'createdAt', 'updatedAt', 'name', 'description', 'studies', 'annotations']
    
    - meta['datasets'][i]['annotations']: list of dicts for each annotation (image-level):
        ['id', 'parentId', 'createdAt', 'createdById', 'updatedAt', 'updatedById', 'modelId', 'StudyInstanceUID', 'SeriesInstanceUID', 
            'SOPInstanceUID', 'labelId', 'annotationNumber', 'height', 'width', 'data', 'note', 'radlexTagIds', 'isImported', 
            'reviewsPositiveCount', 'reviewsNegativeCount']

    - date format: %Y-%m-%dT%H:%M:%S.%fZ
    - for box annotations: x,y are upper left corner 
    - 'SOPInstanceUID': slice id (single image)
    - 'SeriesInstanceUID': series id (single volume) 
    - 'StudyInstanceUID': exam id (multiple volumes)
    - 'height', 'width': image h, w (not annotation)
"""

INCLUDE_AFTER = datetime.strptime('10-08-2020', '%m-%d-%Y')  # annotations made before INCLUDE_AFTER were used to test md.ai
ANNOT_COMMENTS = pd.read_csv('/Mounts/rbg-storage1/datasets/NLST/mdai/annotation_comments_12062020.csv')

def scale_annotations(annotation, annotation_meta):
    '''
    Scale annotation (bounding boxes) to values in [0,1] by dividing by image height and width (annotation_meta)
    '''
    annotation['x'] /= annotation_meta['width']
    annotation['width'] /= annotation_meta['width']
    annotation['y'] /= annotation_meta['height']
    annotation['height'] /= annotation_meta['height']
    return annotation

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_json_path', type = str, help = 'JSON exported from md.ai', default = '/Mounts/rbg-storage1/datasets/NLST/mdai/mdai_mit_project_poBGbqle_annotations_labelgroup_all_2020-11-25-030811.json')
parser.add_argument('--output_json_path', type = str, help = 'Where to export parsed annotations data')
parser.add_argument('--output_is_dataset_obj', action = 'store_true', default = False, help = 'Whether incorporating annotations into dataset json directly')

if __name__ == "__main__":
    args = parser.parse_args()
    annotation_metadata_json = json.load(open(args.annotation_json_path, 'r'))
    annotation_dict =  {}

    USERS = { user['id']: user['name'] for user in annotation_metadata_json['users']  }
    
    for dataset_dict in annotation_metadata_json['datasets']:
        for annotation_meta_dict in tqdm(dataset_dict['annotations']):
            if annotation_meta_dict['data'] is None:
                continue
            date = datetime.strptime(annotation_meta_dict['createdAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
            if not(date > INCLUDE_AFTER):
                continue
            img_id = annotation_meta_dict['SOPInstanceUID']
            series_id = annotation_meta_dict['SeriesInstanceUID']
            exam_id = annotation_meta_dict['StudyInstanceUID']
            if series_id not in annotation_dict:
                annotation_dict[series_id] = defaultdict(list)
            annotation_meta_dict['data'] = scale_annotations(annotation_meta_dict['data'], annotation_meta_dict)
            annotation_meta_dict['data']['user'] = USERS[annotation_meta_dict['createdById']]
            annotation_dict[series_id][img_id].append( annotation_meta_dict['data'] )
     
    for series_id in annotation_dict.keys():
        if series_id in list(ANNOT_COMMENTS['series_uid']):
            if 'FF' in str(ANNOT_COMMENTS[ANNOT_COMMENTS['series_uid'] == series_id]['comments for Peter']):
                for img_id in annotation_dict[series_id].keys():
                    annotation_dict[series_id][img_id] = [ i for i in annotation_dict[series_id][img_id]  if 'fintelmann' in i['user'] ]
    
    if not args.output_is_dataset_obj:
        json.dump(annotation_dict, open(args.output_json_path, 'w'))
    
    else:
        output_json = json.load(open(args.output_json_path, 'r')) 

        for mrn_row in tqdm(output_json):
            for exam_dict in mrn_row['accessions']:
                for series_id, series_dict in exam_dict['image_series'].items():
                    if series_id in annotation_dict:
                        series_dict['annotations'] = annotation_dict[series_id]
                    else:
                        continue
        
        json.dump(output_json, open(args.output_json_path, 'r'))
