import numpy as np
import torch
from torch.utils import data
import warnings
import json, csv
import traceback
from collections import Counter
from sybil.augmentations import get_augmentations
from tqdm import tqdm 
from sybil.serie import Serie
from sybil.datasets.utils import order_slices, fit_to_length, IMG_PAD_PATH, METAFILE_NOTFOUND_ERR, LOAD_FAIL_MSG
from sybil.loaders.image_loaders import OpenCVLoader, DicomLoader 



class SybilDataset(data.Dataset):
    """
    Abstract Object for all Datasets. All datasets have some metadata
    property associated with them, a create_dataset method, a task, and a check
    label and get label function.
    """
    def __init__(self, args, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        '''
        super(SybilDataset, self).__init__()
        
        self.split_group = split_group
        self.args = args
        self._num_images = args.num_images # number of slices in each volume
        self._max_followup = args.max_followup

        try:
            self.dataset_dicts = self.parse_csv_dataset(args.dataset_file_path)
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

        augmentations = get_augmentations(split_group, args)
        if args.img_file_type == 'dicom':
            self.input_loader = DicomLoader(args.cache_path, augmentations, args)  
        else:
            self.input_loader = OpenCVLoader(args.cache_path, augmentations, args)
            
        self.dataset = self.create_dataset(split_group, args.img_dir)
        if len(self.dataset) == 0:
            return
        
        print(self.get_summary_statement(self.dataset, split_group))
        
        dist_key = 'y'
        label_dist = [d[dist_key] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1./ len(label_counts)
        label_weights = {
            label: weight_per_label/count for label, count in label_counts.items()
            }
        
        print("Class counts are: {}".format(label_counts))
        print("Label weights are {}".format(label_weights))
        self.weights = [ label_weights[d[dist_key]] for d in self.dataset]


    def parse_csv_dataset(self, file_path):
        """
        Convert a CSV file into a list of dictionaries for each patient like:
        [
            {
                'patient_id': str, 
                'split': str, 
                'exam_id': str,
                'series_id': str,
                'ever_has_future_cancer': bool
                'years_to_cancer': int,
                'years_to_last_negative_followup': int,
                'paths': [str],
                'slice_locations': [str]
            }
        ]

        Parameters
        ----------
        file_path : str
            path to csv file

        Returns
        -------
        list
            list patient cases in the above structure
        """
        dataset_dicts = {}
        _reader = csv.DictReader(open(file_path,'r'))
        for _row in _reader:
            row = {k.encode('ascii', 'ignore').decode(): v.encode('ascii', 'ignore').decode() for k,v in _row.items()}
            patient_id, exam_id, series_id = row['patient_id'], row['exam_id'], row['series_id']
            unique_id = '{}_{}_{}'.format(patient_id, exam_id, series_id)
            if unique_id in dataset_dicts:
                dataset_dicts[unique_id]['paths'].append(row['file_path'])
                dataset_dicts[unique_id]['slice_locations'].append(row['fileslice_position_path'])
            else:
                dataset_dicts[unique_id] = {
                    'unique_id': unique_id,
                    'patient_id': patient_id, 
                    'exam_id': exam_id,
                    'series_id': series_id,
                    'split': row['split'],
                    'ever_has_future_cancer': row['ever_has_future_cancer'],
                    'years_to_cancer': row['years_to_cancer'],
                    'years_to_last_negative_followup': row['years_to_last_negative_followup'],
                    'paths': [ row['file_path'] ],
                    'slice_locations': [ row['slice_position'] ]
                }
        
        dataset_dicts = list(dataset_dicts.values())
        return dataset_dicts

    def create_dataset(self, split_group):
        """
        Gets the dataset from the paths and labels in the json.
        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
        Returns:
            The dataset as a dictionary with img paths, label, 
            and additional information regarding exam or participant
        """
        
        if self.args.assign_splits:
            np.random.seed(self.args.cross_val_seed)
            self.assign_splits(self.metadata_json)

        dataset = []
        
        for mrn_row in tqdm(self.dataset_dicts, position = 0):
            
            label = mrn_row['ever_has_future_cancer']
            censor_time = 1
            
            series_object = Serie(paths, label, censor_time)
            
            if self.skip_sample(series_object, mrn_row, split_group):
                continue
            
            # put in right order and fit to max length
            paths = order_slices(mrn_row['paths'], mrn_row['slice_locations'])
            paths = fit_to_length(paths, self.num_images)
            # obtain labels
            labels = series_object.get_processed_label(self._max_followup)

            sample = {
                'x': paths,
                'series': series_object,
                'y': labels.y,
                'y_seq': labels.y_seq,
                'y_mask': labels.y_mask,
                'time_at_event': labels.censor_time,
                'exam': mrn_row['unique_id'],
                'patient_id': mrn_row['patient_id']
            }

            dataset.append(sample)

        return dataset

    def skip_sample(self, series_object, row, split_group ):
        if row['split'] != split_group:
            return True
        
        if not series_object.has_label():
            return True
        
        return False
    
    @property
    def PAD_PATH(self):
        return IMG_PAD_PATH

    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed Sybil Cancer Risk {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
        class_balance = Counter([d['y'] for d in dataset])
        exams = set([d['exam'] for d in dataset])
        patients = set([d['patient_id'] for d in dataset])
        statement = summary.format(split_group, len(dataset), len(exams), len(patients), class_balance)
        statement += "\n" + "Censor Times: {}".format( Counter([d['time_at_event'] for d in dataset]))
        return statement

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        try:
            x, mask = self.input_loader.get_images(sample['paths'], sample['additionals'], sample)
            
            item = {
                'x': x,
                'y': sample['y'],
                'y_seq': sample['y_seq'],
                'y_mask': sample['y_mask'],
                'time_at_event': sample['time_at_event'],
                'exam': sample['exam']
                }


            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample['paths'], traceback.print_exc()))  
    
