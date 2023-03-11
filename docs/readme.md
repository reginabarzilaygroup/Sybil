# Sybil Training Parameters

### Train Sybil on NLST

##### 1. DICOM Preprocessing

We use the [DICOM toolkit](https://support.dcmtk.org/docs/dcmj2pnm.html) to convert files from the DICOM format to PNG. For each DICOM file, we convert and save it to a PNG directory while preserving the overall file organization. To do this programmatically, clone [OncoData_Public](https://github.com/yala/OncoData_Public/tree/sybil).

Enter into the `OncoData_Public` directory and run the following commands:

- To convert dicom files to pngs, run the following command:
    
        python scripts/dicom_to_png/dicom_to_png.py \
            --dicom_dir /path/to/dicoms \ 
            --png_dir /path/to/pngs \
            --dcmtk \
            --dicom_types generic \
            --window

- To create a JSON file with metadata for each DICOM file, run the following command:
    
        python scripts/dicom_metadata/dicom_metadata_to_json.py \
        --directory /path/to/dicoms \
        --results_path /path/to/dicom_metadata.json

##### 2. Create NLST dataset file

Run [create_nlst_metadata_json.py](../scripts/data/create_nlst_metadata_json.py) with the appropriate file paths as obtained from NLST. 

- `source_json_path`: is the file obtained from step 1 above
- `output_json_path`: will be the the outputted JSON to be used as `dataset_file_path` argument when training
- `png_path_replace_pattern`: is the pattern to replace in the dicom file paths with the file paths for the PNG files

All other arguments are obtained from external sources.

##### 3. Run train script

```sh 
python train.py \
    --train \
    --dataset nlst \
    --batch_size 3  \
    --gpus 8 \
    --precision 16 \
    --max_followup 6 \
    --img_file_type png \
    --min_num_images 0 \
    --num_images 200 \
    --use_only_thin_cuts_for_ct \
    --slice_thickness_filter 2.5 \
    --resample_pixel_spacing_prob 0.5 \
    --use_annotations \
    --region_annotations_filepath /path/to/annot_dir/annotations_122020.json \
    --dataset_file_path /path/to/json_dir/nlst_dataset.json \
    --img_mean 128.1722 \
    --img_std 87.1849 \
    --img_size 256 256 \
    --num_chan 3 \
    --limit_train_batches 0.5 \
    --limit_val_batches 1.0 \
    --max_epochs 10 \
    --init_lr 3e-5 \
    --lr_decay 0.1 \
    --weight_decay 1e-2 \
    --momentum 0.9 \
    --dropout 0.1 \
    --optimizer adam \
    --patience 5 \
    --tuning_metric c_index \
    --num_workers 3 \
    --profiler simple \
    --num_sanity_val_steps 0 \
    --save_dir /path/to/model_dir/sybil1 \
    --img_dir /path/to/img_dir/nlst-ct-png \
    --results_path /path/to/results_dir/sybil.results \
    --cache_path /path/to/cache_dir \
    > /path/to/log_dir/sybil.txt 2>&1 
```