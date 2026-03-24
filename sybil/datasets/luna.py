import os
import copy
import cc3d
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import traceback, warnings
from collections import Counter
from scipy.ndimage import zoom
import torch.nn.functional as F
from monai.data import MetaTensor
from sybil.datasets.nlst import NLST_Survival_Dataset, CT_ITEM_KEYS
from sybil.datasets.utils import fit_to_length, LOAD_FAIL_MSG, get_boxes_per_frame
import pandas as pd
from sybil.utils.augmentations import (
    PatchAugmentations,
    FullAugmentations,
    random_pad_3d_box,
)

class LUNA(NLST_Survival_Dataset):
    def create_dataset(self, split_group):
        """
        Gets the dataset from the paths and labels in the json.
        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
        Returns:
            The dataset as a dictionary with img paths, label,
            and additional information regarding exam or participant
        """
        self.nodule_counts_csv = pd.read_excel(
            "/data/rbg/shared/datasets/LUNA16/LUNA16/lidc-idr_nodule_counts_6-23-2015.xlsx"
        )

        dataset = []

        if self.args.assign_splits:
            np.random.seed(self.args.cross_val_seed)
            self.assign_splits(self.metadata_json)

        for mrn_row in tqdm(self.metadata_json, ncols=100):
            if self.args.assign_splits and (mrn_row["split"] != split_group):
                continue
            pid, exams = mrn_row["pid"], mrn_row["accessions"]

            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():
                    if self.skip_sample(series_dict, exam_dict, mrn_row):
                        continue

                    sample = self.get_volume_dict(
                        series_id, series_dict, exam_dict, mrn_row
                    )
                    if len(sample) == 0:
                        continue

                    if isinstance(sample, list):
                        if sample[0]["pid"] == "LIDC-IDRI-0123":  # bad data
                            continue
                        dataset.extend(sample)
                    else:
                        if sample["pid"] == "LIDC-IDRI-0123":  # bad data
                            continue
                        dataset.append(sample)

        return dataset

    def skip_sample(self, series_dict, exam_dict, mrn_row):
        slice_thickness = series_dict["slice_thickness"]
        # check if restricting to specific slice thicknesses
        if (self.args.slice_thickness_filter is not None) and (
            (slice_thickness in ["", None])
            or (slice_thickness > self.args.slice_thickness_filter)
            or (slice_thickness < 0)
        ):
            return True

        if series_dict["pixel_spacing"] is None:
            return True

        if len(series_dict["paths"]) < self.args.min_num_images:
            return True

        if ("focal_tversky_loss" in self.args.loss_fns) and (
            series_dict.get("segmentation_path", None) is None
        ):
            return True

        if self.args.sample_from_annotated_only and series_dict["empty_mask"]:
            return True

        return False

    def get_volume_dict(self, series_id, series_dict, exam_dict, mrn_row):
        img_paths = series_dict["paths"]

        slice_locations = series_dict["img_position"]
        pixel_spacing = series_dict["pixel_spacing"] + [series_dict["slice_thickness"]]
        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations, reverse=False
        )

        studyuid = exam_dict["exam"]

        y, y_seq, y_mask, time_at_event = self.get_label()

        if self.args.img_file_type == "dicom":
            sorted_img_paths = [
                p.replace(".png", ".dcm").replace("dicom_pngs", "dicoms")
                for p in sorted_img_paths
            ]

        num_nodules = int(
            self.nodule_counts_csv[
                self.nodule_counts_csv["TCIA Patent ID"] == mrn_row["pid"]
            ].iloc[0]["Total Number of Nodules* "]
        )

        sample = {
            "paths": sorted_img_paths,
            "slice_locations": sorted_slice_locs,
            "y": int(y),
            "time_at_event": time_at_event,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "exam": int(
                "{}{}".format(
                    studyuid.replace(".", "")[-5:],
                    series_id.replace(".", "")[-5:],
                )
            ),  # last 5 of study id + last 5 of series id
            "study": studyuid,
            "series": series_id,
            "pid": mrn_row["pid"],
            "pixel_spacing": pixel_spacing,
            "segmentation_path": series_dict["segmentation_path"],
            "num_nodules": num_nodules,
        }

        if self.args.fit_to_length:
            sample["paths"] = fit_to_length(sorted_img_paths, self.args.num_images)
            sample["slice_locations"] = fit_to_length(
                sorted_slice_locs, self.args.num_images, "<PAD>"
            )

        if self.args.use_annotations:
            # mgh has no annotations, so set everything to zero / false
            sample["volume_annotations"] = np.array([0 for _ in sample["paths"]])
            sample["annotations"] = [
                {"image_annotations": None} for path in sample["paths"]
            ]
        return sample

    def get_label(self):
        y = False
        y_seq = np.zeros(self.args.max_followup)
        time_at_event = self.args.max_followup - 1
        y_mask = np.zeros(self.args.max_followup)
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

    def get_ct_annotations(self, sample):
        # correct empty lists of annotations
        sample["annotations"] = np.load(sample["segmentation_path"])["x"]
        return sample

    def __getitem__(self, index):
        sample = copy.deepcopy(self.dataset[index])
        try:
            return self.process_item(sample)
        except Exception as e:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))
            return None

    def process_item(self, sample):
        if self.args.use_annotations:
            mask = np.load(sample["segmentation_path"])["segmentation"]
            sample["annotations"] = (mask > 0) * 1.0

        try:
            item = {}
            input_dict = self.get_images(sample["paths"], sample)

            x = input_dict["input"]

            item["x"] = x
            item["y"] = sample["y"]
            item["mask"] = (
                mask if self.args.keep_original_mask else (input_dict["mask"] > 0) * 1.0
            )
            item["num_nodules"] = sample["num_nodules"]
            item["target_boxes"] = get_boxes_per_frame(mask)
            for key in CT_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

    @staticmethod
    def set_args(args):
        args.num_classes = args.max_followup

    def get_summary_statement(self, dataset, split_group):
        summary = "Constructed LUNA CT {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
        class_balance = Counter([d["y"] for d in dataset])
        exams = set([d["exam"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group,
            len(dataset),
            len(exams),
            len(patients),
            class_balance,
        )
        statement += "\n" + "Censor Times: {}".format(
            Counter([d["time_at_event"] for d in dataset])
        )
        return statement

    def assign_splits(self, meta):
        for idx in range(len(meta)):
            meta[idx]["split"] = np.random.choice(
                ["train", "dev", "test"], p=self.args.split_probs
            )

class LUNA_Patches(LUNA):
    def __init__(self, args, split_group):
        super().__init__(args, split_group)
        self.augmentations = PatchAugmentations(args, split=split_group)
        self.pad = args.anatomix_pad_size
        self.anatomix_crop_size = args.anatomix_crop_size

    def skip_sample(self, series_dict, exam_dict, mrn_row):
        super().skip_sample(series_dict, exam_dict, mrn_row)
        if series_dict.get("boxes", None) is None:
            return True
        if isinstance(series_dict["boxes"], list) and len(series_dict["boxes"]) == 0:
            return True
        return False

    def get_volume_dict(self, series_id, series_dict, exam_dict, mrn_row):
        img_paths = series_dict["paths"]

        slice_locations = series_dict["img_position"]
        pixel_spacing = series_dict["pixel_spacing"] + [series_dict["slice_thickness"]]
        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations, reverse=False
        )

        studyuid = exam_dict["exam"]

        y, y_seq, y_mask, time_at_event = self.get_label()

        if self.args.img_file_type == "dicom":
            sorted_img_paths = [
                p.replace(".png", ".dcm").replace("dicom_pngs", "dicoms")
                for p in sorted_img_paths
            ]

        num_nodules = int(
            self.nodule_counts_csv[
                self.nodule_counts_csv["TCIA Patent ID"] == mrn_row["pid"]
            ].iloc[0]["Total Number of Nodules* "]
        )
        samples = []
        for boxi, box in enumerate(series_dict["boxes"]):
            samples.append(
                {
                    "paths": sorted_img_paths,
                    "slice_locations": sorted_slice_locs,
                    "y": int(y),
                    "time_at_event": time_at_event,
                    "y_seq": y_seq,
                    "y_mask": y_mask,
                    "vol_exam": int(
                        "{}{}".format(
                            studyuid.replace(".", "")[-5:],
                            series_id.replace(".", "")[-5:],
                        )
                    ),  # last 5 of study id + last 5 of series id
                    "exam": int(
                        "{}{}{}".format(
                            studyuid.replace(".", "")[-5:],
                            series_id.replace(".", "")[-5:],
                            boxi + 1,
                        )
                    ),  # last 5 of study id + last 5 of series id
                    "study": studyuid,
                    "series": series_id,
                    "pid": mrn_row["pid"],
                    "pixel_spacing": pixel_spacing,
                    "segmentation_path": series_dict["segmentation_path"],
                    "num_nodules": num_nodules,
                    "box": box,
                    "predicted_boxes": series_dict.get("predicted_boxes", None),
                }
            )

        return samples

    def process_item(self, sample):
        cache_path = (
            os.path.join(self.args.cache_path, f"{sample['vol_exam']}.pt")
            if self.args.cache_path is not None
            else None
        )
        if (self.args.cache_path is not None) and os.path.exists(cache_path):
            saved_input = torch.load(cache_path, weights_only=False)
            image = saved_input["image"]
            label = saved_input["label"]
            if image.shape[:2] != (512, 512):
                # remove file
                os.remove(cache_path)
                affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))
                mask = np.load(sample["segmentation_path"])["segmentation"]
                label = (mask > 0) * 1.0
                # put frames of label in the last dimension
                label = np.transpose(label, (1, 2, 0))
                slices = [
                    self.input_loader.load_input(p, sample)["input"]
                    for p in sample["paths"]
                ]
                image = np.stack(slices, axis=-1)
                image = MetaTensor(
                    image,
                    affine=affine,
                    dtype=torch.float32,
                )
                label = MetaTensor(
                    label,
                    affine=affine,
                    dtype=torch.float32,
                )
                torch.save({"label": label, "image": image}, cache_path)

        else:
            affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))
            mask = np.load(sample["segmentation_path"])["segmentation"]
            label = (mask > 0) * 1.0
            # put frames of label in the last dimension
            label = np.transpose(label, (1, 2, 0))
            slices = [
                self.input_loader.load_input(p, sample)["input"]
                for p in sample["paths"]
            ]
            image = np.stack(slices, axis=-1)
            image = MetaTensor(
                image,
                affine=affine,
                dtype=torch.float32,
            )
            label = MetaTensor(
                label,
                affine=affine,
                dtype=torch.float32,
            )
            if self.args.cache_path is not None:
                # save the sample to cache
                torch.save({"label": label, "image": image}, cache_path)

        # apply the box to the image and label
        box = sample["box"]

        # Ensure image and label have the correct spatial size (args.img_size)
        img_size = self.args.img_size  # (H, W)
        img_h, img_w = image.shape[0], image.shape[1]
        if (img_h, img_w) != tuple(img_size):
            # image: (H, W, D), label: (H, W, D)
            # Add batch and channel dims for interpolation: (1, 1, D, H, W)
            image_ = image.permute(2, 0, 1).unsqueeze(1)
            label_ = label.permute(2, 0, 1).unsqueeze(1)
            image_ = F.interpolate(
                image_,
                size=(img_size[0], img_size[1]),
                mode="bilinear",
                align_corners=False,
            )
            label_ = F.interpolate(
                label_, size=(img_size[0], img_size[1]), mode="nearest"
            )
            # Remove batch/channel dims and permute back: (H, W, D)
            image = image_.squeeze(1).permute(1, 2, 0)
            label = label_.squeeze(1).permute(1, 2, 0)
            # adjust the box to the new image size
            box["y_start"] = int(box["y_start"] * img_size[0] / img_h)
            box["y_stop"] = int(box["y_stop"] * img_size[0] / img_h)
            box["x_start"] = int(box["x_start"] * img_size[1] / img_w)
            box["x_stop"] = int(box["x_stop"] * img_size[1] / img_w)

        if (self.split_group != "train") and self.args.predict:
            sample["image"] = image[None]
            sample["label"] = label[None]
            y = 1
            item = {
                "x": sample["image"].permute(0, 3, 1, 2),
                "mask": sample["label"].permute(0, 3, 1, 2),
                "y": y,
                "exam": sample["vol_exam"],
                "pid": sample["pid"],
                "vol_exam": sample["vol_exam"],
                "dataset": "luna",
            }

        else:
            h1, w1, _ = self.anatomix_crop_size
            d1 = self.pad[-1]

            y = 1
            if self.split_group in ["train", "dev"]:
                if np.random.uniform(0, 1) < self.args.sample_negative_ratio:
                    # Generate a negative box of the same size as the original box, but at a different location

                    # Load lung mask (assume path is in sample["lung_mask_path"])
                    lung_mask = np.load(
                        "/data/rbg/scratch/lung_ct/luna_lung_mask/sample_{}.npy".format(
                            sample["vol_exam"]
                        )
                    )  # binary mask, shape (H, W, D)
                    if (img_h, img_w) != tuple(img_size):
                        lung_mask = zoom(
                            lung_mask,
                            (1, img_size[0] / img_h, img_size[1] / img_w),
                            order=0,
                        ).transpose(1, 2, 0)
                    else:
                        lung_mask = lung_mask.transpose(1, 2, 0)

                    # Find all lung voxel coordinates
                    lung_coords = np.where(lung_mask > 0)
                    if len(lung_coords[0]) > 0:
                        # Randomly select a point inside the lung mask
                        random_idx = np.random.randint(len(lung_coords[0]))
                        y_start = lung_coords[0][random_idx]
                        x_start = lung_coords[1][random_idx]
                        z_start = lung_coords[2][random_idx]

                        y_stop = y_start + box["y_stop"] - box["y_start"]
                        x_stop = x_start + box["x_stop"] - box["x_start"]
                        z_stop = z_start + box["z_stop"] - box["z_start"]

                        box = {
                            "y_start": y_start,
                            "y_stop": y_stop,
                            "x_start": x_start,
                            "x_stop": x_stop,
                            "z_start": z_start,
                            "z_stop": z_stop,
                        }
                        y = 0

            cbbox = random_pad_3d_box(
                box, image, min_height=h1, min_width=w1, min_depth=d1, random_hw=True
            )
            sample["image"] = image[cbbox]
            sample["label"] = label[cbbox]

            try:
                sample = self.augmentations(sample)
            except Exception as e:
                print(
                    "Error in augmentations for sample {}: {}".format(
                        sample["exam"], box
                    )
                )

            item = {
                "x": sample["image"].permute(0, 3, 1, 2),
                "mask": sample["label"].permute(0, 3, 1, 2),
                "y": y,
                "exam": sample["exam"],
                "pid": sample["pid"],
                "vol_exam": sample["vol_exam"],
                "dataset": "luna",
            }

        return item

class LUNA_Confidence(LUNA_Patches):
    def process_item(self, sample):
        cache_path = (
            os.path.join(self.args.cache_path, f"{sample['vol_exam']}.pt")
            if self.args.cache_path is not None
            else None
        )
        if (self.args.cache_path is not None) and os.path.exists(cache_path):
            saved_input = torch.load(cache_path, weights_only=False)
            image = saved_input["image"]
            label = saved_input["label"]

        else:
            affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))
            mask = np.load(sample["segmentation_path"])["segmentation"]
            label = (mask > 0) * 1.0
            # put frames of label in the last dimension
            label = np.transpose(label, (1, 2, 0))
            slices = [
                self.input_loader.load_input(p, sample)["input"]
                for p in sample["paths"]
            ]
            image = np.stack(slices, axis=-1)
            image = MetaTensor(
                image,
                affine=affine,
                dtype=torch.float32,
            )
            label = MetaTensor(
                label,
                affine=affine,
                dtype=torch.float32,
            )
            if self.args.cache_path is not None:
                # save the sample to cache
                torch.save({"label": label, "image": image}, cache_path)

        # apply the box to the image and label
        box = sample["box"]

        # Ensure image and label have the correct spatial size (args.img_size)
        img_size = self.args.img_size  # (H, W)
        img_h, img_w = image.shape[0], image.shape[1]
        if (img_h, img_w) != tuple(img_size):
            # image: (H, W, D), label: (H, W, D)
            # Add batch and channel dims for interpolation: (1, 1, D, H, W)
            image_ = image.permute(2, 0, 1).unsqueeze(1)
            label_ = label.permute(2, 0, 1).unsqueeze(1)
            image_ = F.interpolate(
                image_,
                size=(img_size[0], img_size[1]),
                mode="bilinear",
                align_corners=False,
            )
            label_ = F.interpolate(
                label_, size=(img_size[0], img_size[1]), mode="nearest"
            )
            # Remove batch/channel dims and permute back: (H, W, D)
            image = image_.squeeze(1).permute(1, 2, 0)
            label = label_.squeeze(1).permute(1, 2, 0)
            # adjust the box to the new image size
            box["y_start"] = int(box["y_start"] * img_size[0] / img_h)
            box["y_stop"] = int(box["y_stop"] * img_size[0] / img_h)
            box["x_start"] = int(box["x_start"] * img_size[1] / img_w)
            box["x_stop"] = int(box["x_stop"] * img_size[1] / img_w)

        if (self.split_group != "train") and self.args.predict:
            sample["image"] = image[None]
            sample["label"] = label[None]
            y = 1
            item = {
                "x": sample["image"].permute(0, 3, 1, 2),
                "mask": sample["label"].permute(0, 3, 1, 2),
                "y": y,
                "exam": sample["vol_exam"],
                "pid": sample["pid"],
                "vol_exam": sample["vol_exam"],
                "dataset": "luna",
            }

        else:
            h1, w1, _ = self.anatomix_crop_size
            d1 = self.pad[-1]

            y = 1
            if self.split_group in ["train", "dev"]:
                sample_from_predictions = (
                    np.random.uniform(0, 1) < self.args.sample_negative_ratio
                )
                use_teacher_forcing = (
                    np.random.uniform(0, 1) < self.args.teacher_force_ratio
                )
                if sample_from_predictions or (not use_teacher_forcing):
                    segmentation = pickle.load(
                        open(
                            f"/data/rbg/scratch/lung_ct/luna-stmix/last/sample_{sample['vol_exam']}.hiddens",
                            "rb",
                        )
                    )
                    segmentation = (segmentation["hidden"][1] > 0.5) * 1.0

                if sample_from_predictions:
                    voxel_scaling = (
                        np.prod(sample["pixel_spacing"])
                        / (img_size[0] / img_h)
                        / (img_size[1] / img_w)
                    )
                    boxes = [
                        (box, voxels)
                        for box, voxels in sample["predicted_boxes"]
                        if voxels * voxel_scaling > 10
                    ]
                    box, voxels = random.sample(boxes, 1)[0]
                    y = int(
                        (
                            label[
                                box["y_start"] : box["y_stop"],
                                box["x_start"] : box["x_stop"],
                                box["z_start"] : box["z_stop"],
                            ].sum()
                            > 0
                        ).item()
                    )

                    label = torch.zeros_like(label)
                    label[
                        box["y_start"] : box["y_stop"],
                        box["x_start"] : box["x_stop"],
                        box["z_start"] : box["z_stop"],
                    ] = segmentation[
                        box["z_start"] : box["z_stop"],
                        box["y_start"] : box["y_stop"],
                        box["x_start"] : box["x_stop"],
                    ].permute(1, 2, 0)
                elif not use_teacher_forcing:
                    label = torch.zeros_like(label)
                    label[
                        box["y_start"] : box["y_stop"],
                        box["x_start"] : box["x_stop"],
                        box["z_start"] : box["z_stop"],
                    ] = segmentation[
                        box["z_start"] : box["z_stop"],
                        box["y_start"] : box["y_stop"],
                        box["x_start"] : box["x_stop"],
                    ].permute(1, 2, 0)

            cbbox = random_pad_3d_box(
                box,
                image,
                min_height=h1,
                min_width=w1,
                min_depth=d1,
                random_hw=False,
                random_d=False,
            )
            sample["image"] = image[cbbox]
            sample["label"] = label[cbbox]

            try:
                sample = self.augmentations(sample)
            except Exception as e:
                print(
                    "Error in augmentations for sample {}: {}".format(
                        sample["exam"], box
                    )
                )

            sample["image"] = sample["image"].permute(0, 3, 1, 2)
            sample["label"] = sample["label"].permute(0, 3, 1, 2)
            # concat image and label
            x = torch.cat([sample["image"], sample["label"]], dim=0)

            item = {
                "x": x,
                "y": y,
                "exam": sample["exam"],
                "pid": sample["pid"],
                "vol_exam": sample["vol_exam"],
                "dataset": "luna",
            }

        return item

class LUNA_Confidence_Flat(LUNA_Patches):
    def get_volume_dict(self, series_id, series_dict, exam_dict, mrn_row):
        img_paths = series_dict["paths"]

        slice_locations = series_dict["img_position"]
        pixel_spacing = series_dict["pixel_spacing"] + [series_dict["slice_thickness"]]
        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations, reverse=False
        )

        studyuid = exam_dict["exam"]

        y, y_seq, y_mask, time_at_event = self.get_label()

        if self.args.img_file_type == "dicom":
            sorted_img_paths = [
                p.replace(".png", ".dcm").replace("dicom_pngs", "dicoms")
                for p in sorted_img_paths
            ]

        num_nodules = int(
            self.nodule_counts_csv[
                self.nodule_counts_csv["TCIA Patent ID"] == mrn_row["pid"]
            ].iloc[0]["Total Number of Nodules* "]
        )
        samples = []
        for boxi, box in enumerate(series_dict["boxes"]):
            samples.append(
                {
                    "paths": sorted_img_paths,
                    "slice_locations": sorted_slice_locs,
                    "y": int(y),
                    "time_at_event": time_at_event,
                    "y_seq": y_seq,
                    "y_mask": y_mask,
                    "vol_exam": int(
                        "{}{}".format(
                            studyuid.replace(".", "")[-5:],
                            series_id.replace(".", "")[-5:],
                        )
                    ),  # last 5 of study id + last 5 of series id
                    "exam": int(
                        "{}{}{}".format(
                            studyuid.replace(".", "")[-5:],
                            series_id.replace(".", "")[-5:],
                            boxi + 1,
                        )
                    ),  # last 5 of study id + last 5 of series id
                    "study": studyuid,
                    "series": series_id,
                    "pid": mrn_row["pid"],
                    "pixel_spacing": pixel_spacing,
                    "segmentation_path": series_dict["segmentation_path"],
                    "num_nodules": num_nodules,
                    "box": box,
                    "is_true_box": True,
                }
            )
        boxi_offset = len(samples)
        pred_boxes = series_dict.get("predicted_boxes", [])
        if pred_boxes is None:
            pred_boxes = []
        for boxi, (box, nvoxel) in enumerate(pred_boxes):
            if nvoxel * np.prod(pixel_spacing) < self.args.min_nodule_volume:
                continue

            samples.append(
                {
                    "paths": sorted_img_paths,
                    "slice_locations": sorted_slice_locs,
                    "y": int(y),
                    "time_at_event": time_at_event,
                    "y_seq": y_seq,
                    "y_mask": y_mask,
                    "vol_exam": int(
                        "{}{}".format(
                            studyuid.replace(".", "")[-5:],
                            series_id.replace(".", "")[-5:],
                        )
                    ),  # last 5 of study id + last 5 of series id
                    "exam": int(
                        "{}{}{}".format(
                            studyuid.replace(".", "")[-5:],
                            series_id.replace(".", "")[-5:],
                            boxi + 1 + boxi_offset,
                        )
                    ),  # last 5 of study id + last 5 of series id
                    "study": studyuid,
                    "series": series_id,
                    "pid": mrn_row["pid"],
                    "pixel_spacing": pixel_spacing,
                    "segmentation_path": series_dict["segmentation_path"],
                    "num_nodules": num_nodules,
                    "box": box,
                    "is_true_box": False,
                }
            )

        return samples

    def process_item(self, sample):
        cache_path = (
            os.path.join(self.args.cache_path, f"{sample['vol_exam']}.pt")
            if self.args.cache_path is not None
            else None
        )
        if (self.args.cache_path is not None) and os.path.exists(cache_path):
            saved_input = torch.load(cache_path, weights_only=False)
            image = saved_input["image"]
            label = saved_input["label"]

        else:
            affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))
            mask = np.load(sample["segmentation_path"])["segmentation"]
            label = (mask > 0) * 1.0
            # put frames of label in the last dimension
            label = np.transpose(label, (1, 2, 0))
            slices = [
                self.input_loader.load_input(p, sample)["input"]
                for p in sample["paths"]
            ]
            image = np.stack(slices, axis=-1)
            image = MetaTensor(
                image,
                affine=affine,
                dtype=torch.float32,
            )
            label = MetaTensor(
                label,
                affine=affine,
                dtype=torch.float32,
            )
            if self.args.cache_path is not None:
                # save the sample to cache
                torch.save({"label": label, "image": image}, cache_path)

        # apply the box to the image and label
        box = sample["box"]
        is_true_box = sample.get("is_true_box", False)

        # Ensure image and label have the correct spatial size (args.img_size)
        img_size = self.args.img_size  # (H, W)
        img_h, img_w = image.shape[0], image.shape[1]
        if (img_h, img_w) != tuple(img_size):
            # image: (H, W, D), label: (H, W, D)
            # Add batch and channel dims for interpolation: (1, 1, D, H, W)
            image_ = image.permute(2, 0, 1).unsqueeze(1)
            label_ = label.permute(2, 0, 1).unsqueeze(1)
            image_ = F.interpolate(
                image_,
                size=(img_size[0], img_size[1]),
                mode="bilinear",
                align_corners=False,
            )
            label_ = F.interpolate(
                label_, size=(img_size[0], img_size[1]), mode="nearest"
            )
            # Remove batch/channel dims and permute back: (H, W, D)
            image = image_.squeeze(1).permute(1, 2, 0)
            label = label_.squeeze(1).permute(1, 2, 0)
            # adjust the box to the new image size
            if is_true_box:
                box["y_start"] = int(box["y_start"] * img_size[0] / img_h)
                box["y_stop"] = int(box["y_stop"] * img_size[0] / img_h)
                box["x_start"] = int(box["x_start"] * img_size[1] / img_w)
                box["x_stop"] = int(box["x_stop"] * img_size[1] / img_w)

        if (self.split_group != "train") and self.args.predict:
            sample["image"] = image[None]
            sample["label"] = label[None]
            y = 1
            item = {
                "x": sample["image"].permute(0, 3, 1, 2),
                "mask": sample["label"].permute(0, 3, 1, 2),
                "y": y,
                "exam": sample["vol_exam"],
                "pid": sample["pid"],
                "vol_exam": sample["vol_exam"],
                "dataset": "luna",
            }

        else:
            h1, w1, _ = self.anatomix_crop_size
            d1 = self.pad[-1]

            y = 1
            if self.split_group in ["train", "dev"]:
                if not is_true_box:
                    segmentation = pickle.load(
                        open(
                            f"/data/rbg/scratch/lung_ct/luna-stmix/last/sample_{sample['vol_exam']}.hiddens",
                            "rb",
                        )
                    )
                    segmentation = (segmentation["hidden"][1] > 0.5) * 1.0

                    y = int(
                        (
                            label[
                                box["y_start"] : box["y_stop"],
                                box["x_start"] : box["x_stop"],
                                box["z_start"] : box["z_stop"],
                            ].sum()
                            > 0
                        ).item()
                    )

                    label = torch.zeros_like(label)
                    label[
                        box["y_start"] : box["y_stop"],
                        box["x_start"] : box["x_stop"],
                        box["z_start"] : box["z_stop"],
                    ] = segmentation[
                        box["z_start"] : box["z_stop"],
                        box["y_start"] : box["y_stop"],
                        box["x_start"] : box["x_stop"],
                    ].permute(1, 2, 0)

            cbbox = random_pad_3d_box(
                box,
                image,
                min_height=h1,
                min_width=w1,
                min_depth=d1,
                random_hw=False,
                random_d=False,
            )
            sample["image"] = image[cbbox]
            sample["label"] = label[cbbox]

            try:
                sample = self.augmentations(sample)
            except Exception as e:
                print(
                    "Error in augmentations for sample {}: {}".format(
                        sample["exam"], box
                    )
                )

            sample["image"] = sample["image"].permute(0, 3, 1, 2)
            sample["label"] = sample["label"].permute(0, 3, 1, 2)
            # concat image and label
            x = torch.cat([sample["image"], sample["label"]], dim=0)

            item = {
                "x": x,
                "y": y,
                "exam": sample["exam"],
                "pid": sample["pid"],
                "vol_exam": sample["vol_exam"],
                "dataset": "luna",
            }

        return item

class LUNA2(LUNA):
    def __init__(self, args, split_group):
        super().__init__(args, split_group)
        self.augmentations = FullAugmentations(args, split=split_group)

    def skip_sample(self, series_dict, exam_dict, mrn_row):
        slice_thickness = series_dict["slice_thickness"]
        # check if restricting to specific slice thicknesses
        if (self.args.slice_thickness_filter is not None) and (
            (slice_thickness in ["", None])
            or (slice_thickness > self.args.slice_thickness_filter)
            or (slice_thickness < 0)
        ):
            return True

        if series_dict["pixel_spacing"] is None:
            return True

        if len(series_dict["paths"]) < self.args.min_num_images:
            return True

        if series_dict.get("segmentation_path", None) is None:
            return True

        if series_dict["empty_mask"]:
            return True

        return False

    def get_volume_dict(self, series_id, series_dict, exam_dict, mrn_row):
        img_paths = series_dict["paths"]

        slice_locations = series_dict["img_position"]
        pixel_spacing = series_dict["pixel_spacing"] + [series_dict["slice_thickness"]]
        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations, reverse=False
        )

        studyuid = exam_dict["exam"]

        y, y_seq, y_mask, time_at_event = self.get_label()

        if self.args.img_file_type == "dicom":
            sorted_img_paths = [
                p.replace(".png", ".dcm").replace("dicom_pngs", "dicoms")
                for p in sorted_img_paths
            ]

        num_nodules = int(
            self.nodule_counts_csv[
                self.nodule_counts_csv["TCIA Patent ID"] == mrn_row["pid"]
            ].iloc[0]["Total Number of Nodules* "]
        )

        sample = {
            "paths": sorted_img_paths,
            "slice_locations": sorted_slice_locs,
            "y": int(y),
            "time_at_event": time_at_event,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "exam": int(
                "{}{}".format(
                    studyuid.replace(".", "")[-5:],
                    series_id.replace(".", "")[-5:],
                )
            ),  # last 5 of study id + last 5 of series id
            "study": studyuid,
            "series": series_id,
            "pid": mrn_row["pid"],
            "pixel_spacing": pixel_spacing,
            "segmentation_path": series_dict["segmentation_path"],
            "num_nodules": num_nodules,
            "has_y": 0,
            "has_mask": 1,
        }

        if self.args.fit_to_length:
            sample["paths"] = fit_to_length(sorted_img_paths, self.args.num_images)
            sample["slice_locations"] = fit_to_length(
                sorted_slice_locs, self.args.num_images, "<PAD>"
            )

        if self.args.use_annotations:
            # mgh has no annotations, so set everything to zero / false
            sample["volume_annotations"] = np.array([0 for _ in sample["paths"]])
            sample["annotations"] = [
                {"image_annotations": None} for path in sample["paths"]
            ]
        return sample

    def process_item(self, sample):
        cache_path = (
            os.path.join(self.args.cache_path, f"{sample['exam']}.pt")
            if self.args.cache_path is not None
            else None
        )
        if (self.args.cache_path is not None) and os.path.exists(cache_path):
            saved_input = torch.load(cache_path, weights_only=False)
            image = saved_input["image"]
            label = saved_input["label"]

        else:
            affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))
            mask = np.load(sample["segmentation_path"])["segmentation"]
            label = (mask > 0) * 1.0
            # put frames of label in the last dimension
            label = np.transpose(label, (1, 2, 0))
            slices = [
                self.input_loader.load_input(p, sample)["input"]
                for p in sample["paths"]
            ]
            image = np.stack(slices, axis=-1)
            image = MetaTensor(
                image,
                affine=affine,
                dtype=torch.float32,
            )
            label = MetaTensor(
                label,
                affine=affine,
                dtype=torch.float32,
            )
            if self.args.cache_path is not None:
                # save the sample to cache
                torch.save({"label": label, "image": image}, cache_path)

        if not os.path.exists(
            "/data/rbg/scratch/lung_ct/luna_lung_mask/sample_{}.npy".format(
                sample["exam"]
            )
        ):
            return
        lung_mask = np.load(
            "/data/rbg/scratch/lung_ct/luna_lung_mask/sample_{}.npy".format(
                sample["exam"]
            )
        )
        lung_mask = (lung_mask > 0) * 1.0
        lung_mask = MetaTensor(
            np.transpose(lung_mask, (1, 2, 0)),
            affine=torch.diag(torch.tensor(sample["pixel_spacing"] + [1])),
            dtype=torch.float32,
        )
        image_dict = {
            "image": image,
            "label": label,
            "lung": lung_mask,
        }
        image_dict = self.augmentations(image_dict)
        image_dict["label"] = self.augmentations.segmentation_transform(label)
        label = (image_dict["label"].permute(0, 3, 1, 2) > 0) * 1.0
        label, num_nodules = cc3d.connected_components(label[0], return_N=True)

        item = {
            "x": image_dict["image"].permute(0, 3, 1, 2),
            "mask": label[None],
            "lung": image_dict["lung"].permute(0, 3, 1, 2),
            "y": sample["y"],
            "has_y": sample["has_y"],
            "has_mask": sample["has_mask"],
            "cancer_laterality": 0,
            "has_laterality_class": 0,
            "exam": sample["exam"],
            "pid": sample["pid"],
            "dataset": "luna",
            "num_nodules": num_nodules,
            "nodule_ids": torch.arange(1, num_nodules + 1),
            "nodule_has_cancer_labels": [0],
            "nodule_has_luna25_labels": [0],
            "nodule_volumes": [0],
            "nodule_cancer_labels": [0],
            "nodule_luna25_labels": [0],
            "time_at_event": sample["time_at_event"],
            "y_seq": sample["y_seq"],
            "y_mask": sample["y_mask"],
        }
        return item

    def get_summary_statement(self, dataset, split_group):
        summary = "Constructed LUNA CT {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
        class_balance = Counter([d["y"] for d in dataset])
        exams = set([d["exam"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group,
            len(dataset),
            len(exams),
            len(patients),
            class_balance,
        )
        statement += "\n" + "Censor Times: {}".format(
            Counter([d["time_at_event"] for d in dataset])
        )
        return statement
