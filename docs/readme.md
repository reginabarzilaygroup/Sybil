# Sybil Training Parameters

### Train Sybil on NLST

```sh 
python train.py \
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