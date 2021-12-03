import json
import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd 
import time
from collections import defaultdict
from ast import literal_eval
SPLIT_PROBS = [0.8, 0.2]


parser = argparse.ArgumentParser()
parser.add_argument('--output_json_path', type = str, default = '/Mounts/rbg-storage1/datasets/NLST/full_nlst.json')
parser.add_argument('--source_json_path', type = str, default = '/Mounts/rbg-storage1/datasets/NLST/nlst_metadata_022020.json')
parser.add_argument('--nlst_abnormalities_csv', type = str, default = '/Mounts/rbg-storage1/datasets/NLST/package-nlst-564.2020-01-30/NLST_564/nlst_564.delivery.010220/nlst_564_ct_ab_20191001.csv')
parser.add_argument('--nlst_metadata_csv', type = str, default = '/Mounts/rbg-storage1/datasets/NLST/package-nlst-564.2020-01-30/NLST_564/nlst_564.delivery.010220/nlst_564_prsn_20191001.csv')
parser.add_argument('--nlst_imagedata_csv', type = str, default = '/Mounts/rbg-storage1/datasets/NLST/package-nlst-564.2020-01-30/NLST_564/nlst_564.delivery.010220/nlst_564_ct_image_info_20191001.csv')
parser.add_argument('--test_google_splits', type = str, default = '/Mounts/rbg-storage1/datasets/NLST/Shetty_et_al(Google)/TEST_41591_2019_447_MOESM5_ESM.xlsx')

if __name__ == '__main__':
    args = parser.parse_args()

    # Google Splits
    test_google_data = pd.read_excel(args.test_google_splits)
    test_google_pids = [str(p) for p in test_google_data['patient_id']]

    # Source json, output of OncoData .. 
    source_json = json.load(open(args.source_json_path, 'r'))

    # Dataset json, create new or update existing
    if os.path.exists(args.output_json_path):
        json_dataset = json.load(open(args.output_json_path, 'r'))
        pid_list = [d['pid'] for d in  json_dataset ]
    else:
        json_dataset, pid_list = [], []
    processed_len = len(pid_list)

    abnormalities_data = pd.read_csv(args.nlst_abnormalities_csv, low_memory = True)
    image_data = pd.read_csv(args.nlst_imagedata_csv, low_memory = True)
    meta_data = pd.read_csv(args.nlst_metadata_csv, low_memory = True)

    abnormalities_data.fillna(-1, inplace = True)
    image_data.fillna(-1, inplace = True)
    meta_data.fillna(-1, inplace = True)


    def make_metadata_dict(dataframe, pid, timepoint, series_id, use_timepoint = False, use_timepoint_and_studyinstance = False):
        if use_timepoint_and_studyinstance:
            df = dataframe.loc[(dataframe.pid == int(pid)) & (dataframe.study_yr == timepoint ) & (dataframe.seriesinstanceuids == series_id )]
        elif use_timepoint:
            df = dataframe.loc[(dataframe.pid == int(pid)) & (dataframe.study_yr == timepoint )]
        else:
            df = dataframe.loc[(dataframe.pid == int(pid))]

        if df.shape[0] > 0:
            return df.to_dict('list')
        else:
            return {}

    now = time.time()
    processed_dict = defaultdict(list)

    # accession, pid, scanner, year, slicelocations 
    for row_dict in tqdm(source_json):
        dcm_keys = list(row_dict['dicom_metadata'].keys())
        if ('PatientID' not in dcm_keys) or ('StudyDate' not in dcm_keys) or ('AccessionNumber' not in dcm_keys) or ('ClinicalTrialTimePointID' not in dcm_keys) or ('SliceLocation' not in dcm_keys):
            continue
        pid = row_dict['dicom_metadata']['PatientID']
        date = row_dict['dicom_metadata']['StudyDate']
        accession_number = row_dict['dicom_metadata']['AccessionNumber']
        slicelocation = float(row_dict['dicom_metadata']['SliceLocation'])
        img_posn = float(literal_eval(row_dict['dicom_metadata']['ImagePositionPatient'])[-1])
        timepoint = int(row_dict['dicom_metadata']['ClinicalTrialTimePointID'][-1]) # convert from 'T1' to 1
        exam = '{}_T{}'.format(accession_number, timepoint)
        series_id = row_dict['dicom_metadata']['SeriesInstanceUID']
        slicenumber = int(row_dict['dicom_metadata']['InstanceNumber'])
        pixel_spacing = list(map(float, eval(row_dict['dicom_metadata']['PixelSpacing'])))
        slice_thickness = float(row_dict['dicom_metadata']['SliceThickness']) 
        path = row_dict['dicom_path'].replace('/data/rsg/mammogram', '/Mounts/rbg-storage1/datasets').replace('nlst-ct', 'nlst-ct-png') + '.png'

        # in case resuming from previous time point - skip over processed (patient, exam, series, path)

        if len(processed_dict['pids']) < processed_len:
            if (pid not in processed_dict['pids']): 
                processed_dict['pids'].append(pid)
                if len(processed_dict['pids'])%1200==0: 
                    elapsed = round((time.time()-now)/3600, 3)
                    print('Skipped {} processed participants in {} hrs'.format(len(processed_dict['pids']), elapsed))
            processed_dict['exams'].append(exam)
            processed_dict['dcms'].append(path)
            continue
        
        if (len(processed_dict['pids']) == len(pid_list)) and (pid in pid_list):
            pt_idx = pid_list.index(pid)
            existing_exams = [ exam['exam'] for exam in json_dataset[pt_idx]['accessions'] ]
            if exam in existing_exams:
                exam_idx = existing_exams.index(exam)
                if (series_id in list(json_dataset[pt_idx]['accessions'][exam_idx]['image_series'].keys()) ) and (path in json_dataset[pt_idx]['accessions'][exam_idx]['image_series'][series_id]['paths']):
                    processed_dict['exams'].append(exam)
                    processed_dict['dcms'].append(path)
                    continue
        
        if len(pid_list) == len(set(processed_dict['pids'])) and (pid not in pid_list):
            print('Already processed {} participants, with {} exams, and {} dicoms'.format(len(pid_list), len(set(processed_dict['exams'])), len(set(processed_dict['dcms']))))
        

        exam_dict = {
            'exam': exam,
            'accession_number': accession_number,
            'screen_timepoint': timepoint,
            'date': date,
            'abnormalities': make_metadata_dict(abnormalities_data, pid, timepoint, series_id, use_timepoint = True),
            }

        img_series_dict = {
            'paths': [path],
            'slice_location': [slicelocation],
            'slice_number': [slicenumber],
            'pixel_spacing': pixel_spacing,
            'slice_thickness': slice_thickness,
            'img_position': img_posn,
            'series_data': make_metadata_dict(image_data, pid, timepoint, series_id, use_timepoint_and_studyinstance = True)
            }
                
        if pid in pid_list:
            pt_idx = pid_list.index(pid)
            existing_exams = [ exam['exam'] for exam in json_dataset[pt_idx]['accessions'] ]
            if exam in existing_exams:
                exam_idx = existing_exams.index(exam)
                if series_id not in list(json_dataset[pt_idx]['accessions'][exam_idx]['image_series'].keys()):
                    json_dataset[pt_idx]['accessions'][exam_idx]['image_series'][series_id] = img_series_dict
                else:
                    json_dataset[pt_idx]['accessions'][exam_idx]['image_series'][series_id]['paths'].append(path)
                    json_dataset[pt_idx]['accessions'][exam_idx]['image_series'][series_id]['slice_location'].append(slicelocation)
                    json_dataset[pt_idx]['accessions'][exam_idx]['image_series'][series_id]['slice_number'].append(slicenumber)
            else:
                exam_dict['image_series'] = {series_id: img_series_dict}
                json_dataset[pt_idx]['accessions'].append(exam_dict)
        
        else:
            exam_dict['image_series'] = {series_id: img_series_dict}
            if pid in test_google_pids:
                split_group = 'test'
            else:
                split_group = np.random.choice(['train', 'dev'], p = SPLIT_PROBS)
            pt_dict = {
            'accessions': [exam_dict], 
            'pid': pid, 
            'split': split_group,
            'pt_metadata': make_metadata_dict(meta_data, pid, timepoint, series_id)
            }

            json_dataset.append(pt_dict)
            if (len(pid_list) > 0) and (len(pid_list)%500 == 0):
                elapsed = round((time.time()-now)/3600, 3)
                print('Processed {} participants in {} hrs'.format(len(pid_list), elapsed))
            pid_list.append(pid)
    json.dump(json_dataset, open(args.output_json_path, 'w'))
