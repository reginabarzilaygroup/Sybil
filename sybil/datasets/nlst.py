import os
import json
import copy
import torch
import pickle
import pydicom
import random
import numpy as np
import torchio as tio
from tqdm import tqdm
import traceback, warnings
from torch.utils import data
import torch.nn.functional as F
from collections import Counter, defaultdict
from monai.data import MetaTensor
from sybil.utils.loading import get_sample_loader
from sybil.utils.dicom_to_nifti import pydicom_to_nifti
from sybil.loaders.image_loaders import apply_windowing, apply_pillar_windowing
from sybil.datasets.utils import (
    load_nifti,
    fit_to_length,
    DEVICE_ID,
    LOAD_FAIL_MSG,
    METAFILE_NOTFOUND_ERR,
)
from sybil.utils.augmentations import (
    PatchAugmentations,
    FullAugmentations,
    PillarAugmentations,
    random_pad_3d_box,
)
import SimpleITK as sitk
import pandas as pd
import cc3d
import rve

warnings.filterwarnings("ignore")

GOOGLE_SPLITS_FILENAME = "/data/rbg/shared/projects/sybil/google_data_splits.p"

CORRUPTED_PATHS = "/data/rbg/shared/datasets/NLST/NLST/corrupted_img_paths.pkl"

CT_ITEM_KEYS = [
    "pid",
    "exam",
    "series",
    "y_seq",
    "y_mask",
    "time_at_event",
    "cancer_laterality",
    "has_annotation",
    "origin_dataset",
    "device",
    "slice_thickness",
]

RACE_ID_KEYS = {
    1: "white",
    2: "black",
    3: "asian",
    4: "american_indian_alaskan",
    5: "native_hawaiian_pacific",
    6: "hispanic",
}
ETHNICITY_KEYS = {1: "Hispanic or Latino", 2: "Neither Hispanic nor Latino"}
GENDER_KEYS = {1: "Male", 2: "Female"}
EDUCAT_LEVEL = {
    1: 1,  # 8th grade = less than HS
    2: 1,  # 9-11th = less than HS
    3: 2,  # HS Grade
    4: 3,  # Post-HS
    5: 4,  # Some College
    6: 5,  # Bachelors = College Grad
    7: 6,  # Graduate School = Postrad/Prof
}

STAGE_MAP = {
    110: 0,  # Stage IA
    120: 1,  # Stage IB
    210: 2,  # Stage IIA
    220: 3,  # Stage IIB
    310: 4,  # Stage IIIA
    320: 5,  # Stage IIIB
    400: 6,  # Stage IV
    900: 7,  # Occult Carcinoma
}

DEVICE_TO_NAME = {
    1: "GE MEDICAL SYSTEMS",
    2: "Philips",
    3: "SIEMENS",
    4: "TOSHIBA",
    "GE MEDICAL SYSTEMS": "GE MEDICAL SYSTEMS",
    "Philips": "Philips",
    "SIEMENS": "SIEMENS",
    "TOSHIBA": "TOSHIBA",
    "GE MEDICAL S]STEMS": "GE MEDICAL SYSTEMS",
}

ANATOMICAL_WINDOWS = {
    "CT": {
        "lung": {"center": -600, "width": 1500},
        "mediastinum": {"center": 50, "width": 400},
        "abdomen": {"center": 40, "width": 400},
        "liver": {"center": 80, "width": 150},
        "bone": {"center": 400, "width": 1800},
        "brain": {"center": 40, "width": 80},
        "subdural": {"center": 75, "width": 215},
        "stroke": {"center": 40, "width": 40},
        "temporal_bone": {"center": 600, "width": 2800},
        "soft_tissue": {"center": 50, "width": 350},
    }
}

PID2DICOM_DIRECTORY = pickle.load(
    open("/data/rbg/shared/datasets/NLST/NLST/all_nlst_dicoms_pid2directory.p", "rb")
)


def get_examid(pid, timepoint, series_id):
    return "{}{}{}{}".format(
        pid,
        timepoint,
        series_id.split(".")[-1][:5],
        series_id.split(".")[-1][-5:],
    )



class NLST_Survival_Dataset(data.Dataset):
    def __init__(self, args, split_group):
        """
        NLST Dataset
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(NLST_Survival_Dataset, self).__init__()

        self.split_group = split_group
        self.args = args
        self._num_images = args.num_images  # number of slices in each volume
        self._max_followup = args.max_followup

        try:
            self.metadata_json = json.load(open(args.dataset_file_path, "r"))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

        self.input_loader = get_sample_loader(split_group, args)
        self.always_resample_pixel_spacing = (args.resample_pixel_spacing) and (
            split_group in ["dev", "test"]
        )
        if args.resample_pixel_spacing:
            img_size = args.img_size
            self.resample_transform = tio.transforms.Resample(
                target=tuple(args.ct_pixel_spacing)
            )
            self.padding_transform = tio.transforms.CropOrPad(
                target_shape=tuple(img_size + [args.num_images]), padding_mode=0
            )
        self.pad = not getattr(args, "resample_without_pad", False)

        if self.args.region_annotations_filepath:
            self.annotations_metadata = json.load(
                open(self.args.region_annotations_filepath, "r")
            )
        else:
            self.annotations_metadata = {}

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        print(self.get_summary_statement(self.dataset, split_group))

        if args.class_bal:
            label_dist = [d[args.class_bal_key] for d in self.dataset]
            label_counts = Counter(label_dist)
            weight_per_label = 1.0 / len(label_counts)
            label_weights = {
                label: weight_per_label / count for label, count in label_counts.items()
            }

            print("Class counts are: {}".format(label_counts))
            print("Label weights are {}".format(label_weights))
            self.weights = [label_weights[d[args.class_bal_key]] for d in self.dataset]
        self.metadata_json = None

    def create_dataset(self, split_group):
        """
        Gets the dataset from the paths and labels in the json.
        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
        Returns:
            The dataset as a dictionary with img paths, label,
            and additional information regarding exam or participant
        """

        self.corrupted_paths = self.CORRUPTED_PATHS["paths"]
        self.corrupted_series = self.CORRUPTED_PATHS["series"]
        

        if self.args.assign_splits:
            np.random.seed(self.args.cross_val_seed)
            self.assign_splits(self.metadata_json)

        dataset = []

        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row.get("split", "train"),
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            if (split != split_group) and not self.args.turn_off_splits:
                continue

            for exam_dict in exams:
                if self.args.turn_off_splits:
                    thinnest_series_id = self.get_thinnest_cut(exam_dict)
                else:
                    if self.args.use_only_thin_cuts_for_ct and split_group in [
                        "train",
                        "dev",
                    ]:
                        thinnest_series_id = self.get_thinnest_cut(exam_dict)

                    elif split == "test" and (self.args.assign_splits):
                        thinnest_series_id = self.get_thinnest_cut(exam_dict)

                    elif split == "test":
                        google_series = list(self.GOOGLE_SPLITS[pid]["exams"])
                        nlst_series = list(exam_dict["image_series"].keys())
                        thinnest_series_id = [
                            s for s in nlst_series if s in google_series
                        ]
                        assert len(thinnest_series_id) < 2
                        if len(thinnest_series_id) > 0:
                            thinnest_series_id = thinnest_series_id[0]
                        elif len(thinnest_series_id) == 0:
                            if self.args.assign_splits:
                                thinnest_series_id = self.get_thinnest_cut(exam_dict)
                            else:
                                continue

                for series_id, series_dict in exam_dict["image_series"].items():
                    series_dict["series_data"] = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in series_dict["series_data"].items()
                    }
                    if self.skip_sample(series_id, series_dict, exam_dict, pt_metadata):
                        continue

                    if self.args.use_only_thin_cuts_for_ct and (
                        not series_id == thinnest_series_id
                    ):
                        continue

                    sample = self.get_volume_dict(
                        series_id, series_dict, exam_dict, pt_metadata, pid, split
                    )
                    if len(sample) == 0:
                        continue

                    if isinstance(sample, list):
                        dataset.extend(sample)
                    else:
                        dataset.append(sample)

        return dataset

    def get_thinnest_cut(self, exam_dict):
        # volume that is not thin cut might be the one annotated; or there are multiple volumes with same num slices, so:
        # use annotated if available, otherwise use thinnest cut
        possibly_annotated_series = [
            s in self.annotations_metadata
            for s in list(exam_dict["image_series"].keys())
        ]
        series_lengths = [
            len(exam_dict["image_series"][series_id]["paths"])
            for series_id in exam_dict["image_series"].keys()
        ]
        thinnest_series_len = max(series_lengths)
        thinnest_series_id = [
            k
            for k, v in exam_dict["image_series"].items()
            if len(v["paths"]) == thinnest_series_len
        ]
        if any(possibly_annotated_series):
            thinnest_series_id = list(exam_dict["image_series"].keys())[
                possibly_annotated_series.index(1)
            ]
        else:
            thinnest_series_id = thinnest_series_id[0]
        return thinnest_series_id

    def skip_sample(self, series_id, series_dict, exam_dict, pt_metadata):
        # check if image position is known
        missing_img_position = False
        if None in series_dict["img_position"]:
            missing_img_position = True

        series_data = series_dict["series_data"]
        if "reconthickness" not in series_data:
            slice_thickness = series_dict["slice_thickness"]
            screen_timepoint = exam_dict["screen_timepoint"]
            is_localizer = False
        elif len(series_data):
            # check if screen is localizer screen or not enough images
            is_localizer = self.is_localizer(series_data)

            # check if restricting to specific slice thicknesses
            slice_thickness = series_data["reconthickness"][0]
            screen_timepoint = series_data["study_yr"][0]

        wrong_thickness = (self.args.slice_thickness_filter is not None) and (
            slice_thickness > self.args.slice_thickness_filter or (slice_thickness < 0)
        )

        # check if valid label (info is not missing)

        bad_label = not self.check_label(pt_metadata, screen_timepoint)

        # invalid label
        if not bad_label:
            y, _, _, time_at_event = self.get_label(pt_metadata, screen_timepoint)
            invalid_label = (y == -1) or (time_at_event < 0)
        else:
            invalid_label = False

        insufficient_slices = len(series_dict["paths"]) < self.args.min_num_images

        if (
            is_localizer
            or wrong_thickness
            or bad_label
            or invalid_label
            or insufficient_slices
            or missing_img_position
        ):
            return True
        else:
            return False

    def get_volume_dict(
        self, series_id, series_dict, exam_dict, pt_metadata, pid, split
    ):
        img_paths = series_dict["paths"]
        slice_locations = series_dict["img_position"]
        series_data = series_dict["series_data"]
        device = DEVICE_ID[DEVICE_TO_NAME[series_data["manufacturer"][0]]]
        screen_timepoint = series_data["study_yr"][0]
        assert screen_timepoint == exam_dict["screen_timepoint"]

        if series_id in self.corrupted_series:
            if any([path in self.corrupted_paths for path in img_paths]):
                uncorrupted_imgs = np.where(
                    [path not in self.corrupted_paths for path in img_paths]
                )[0]
                img_paths = np.array(img_paths)[uncorrupted_imgs].tolist()
                slice_locations = np.array(slice_locations)[uncorrupted_imgs].tolist()

        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations
        )

        if not sorted_img_paths[0].startswith(self.args.img_dir):
            sorted_img_paths = [
                self.args.img_dir
                + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                for path in sorted_img_paths
            ]

        if sorted_img_paths[0].endswith(".dcm.png"):
            sorted_img_paths = [p.replace(".dcm.png", ".png") for p in sorted_img_paths]

        if (
            self.args.img_file_type == "dicom"
        ):  # ! NOTE: removing file extension affects get_ct_annotations mapping path to annotation
            sorted_img_paths = [
                (
                    PID2DICOM_DIRECTORY[pid]
                    + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                ).replace(".png", ".dcm")
                for path in sorted_img_paths
            ]

        y, y_seq, y_mask, time_at_event = self.get_label(pt_metadata, screen_timepoint)

        exam_int = int(
            "{}{}{}".format(
                pid, int(screen_timepoint), int(series_id.split(".")[-1][-3:])
            )
        )
        sample = {
            "paths": sorted_img_paths,
            "slice_locations": sorted_slice_locs,
            "y": int(y),
            "time_at_event": time_at_event,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
            "exam": exam_int,
            "accession": exam_dict["accession_number"],
            "series": series_id,
            "study": series_data["studyuid"][0],
            "screen_timepoint": screen_timepoint,
            "pid": pid,
            "device": device,
            "institution": pt_metadata["cen"][0],
            "cancer_laterality": self.get_cancer_side(pt_metadata),
            "num_original_slices": len(series_dict["paths"]),
            "pixel_spacing": series_dict["pixel_spacing"]
            + [series_dict["slice_thickness"]],
            "slice_thickness": self.get_slice_thickness_class(
                series_dict["slice_thickness"]
            ),
        }

        if self.args.use_risk_factors:
            sample["risk_factors"] = self.get_risk_factors(
                pt_metadata, screen_timepoint, return_dict=False
            )

        if self.args.fit_to_length:
            sample["paths"] = fit_to_length(sorted_img_paths, self.args.num_images)
            sample["slice_locations"] = fit_to_length(
                sorted_slice_locs, self.args.num_images, "<PAD>"
            )

        return sample

    def check_label(self, pt_metadata, screen_timepoint):
        valid_days_since_rand = (
            pt_metadata["scr_days{}".format(screen_timepoint)][0] > -1
        )
        valid_days_to_cancer = pt_metadata["candx_days"][0] > -1
        valid_followup = pt_metadata["fup_days"][0] > -1
        return (valid_days_since_rand) and (valid_days_to_cancer or valid_followup)

    def get_label(self, pt_metadata, screen_timepoint):
        max_followup = (
            self.args.max_followup_data
            if self.args.max_followup_data is not None
            else self.args.max_followup
        )
        days_since_rand = pt_metadata["scr_days{}".format(screen_timepoint)][0]
        days_to_cancer_since_rand = pt_metadata["candx_days"][0]
        days_to_cancer = days_to_cancer_since_rand - days_since_rand
        years_to_cancer = (
            int(days_to_cancer // 365) if days_to_cancer_since_rand > -1 else 100
        )
        days_to_last_followup = int(pt_metadata["fup_days"][0] - days_since_rand)
        years_to_last_followup = days_to_last_followup // 365
        y = years_to_cancer < max_followup
        y_seq = np.zeros(max_followup)
        cancer_timepoint = pt_metadata["cancyr"][0]
        if y:
            if years_to_cancer > -1:
                assert screen_timepoint <= cancer_timepoint
            time_at_event = years_to_cancer
            y_seq[years_to_cancer:] = 1
        else:
            time_at_event = min(years_to_last_followup, max_followup - 1)
        y_mask = np.array(
            [1] * (time_at_event + 1) + [0] * (max_followup - (time_at_event + 1))
        )
        assert len(y_mask) == max_followup
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

    def is_localizer(self, series_dict):
        is_localizer = (
            (series_dict.get("imageclass", [-1])[0] == 0)
            or ("LOCALIZER" in series_dict["imagetype"][0])
            or ("TOP" in series_dict["imagetype"][0])
        )
        return is_localizer

    def get_pixel_spacing(self, dcm_path):
        """Get slice thickness and row/col spacing

        Args:
            path (str): path to sample png file in the series

        Returns:
            pixel spacing: [thickness, spacing[0], spacing[1]]
                thickness (float): CT slice thickness
                spacing (list): spacing along x and y axes
        """
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        spacing = [float(d) for d in dcm.PixelSpacing] + [float(dcm.SliceThickness)]
        return spacing

    def get_cancer_side(self, pt_metadata):
        """
        Return if cancer in left or right

        right: (rhil, right hilum), (rlow, right lower lobe), (rmid, right middle lobe), (rmsb, right main stem), (rup, right upper lobe),
        left: (lhil, left hilum),  (llow, left lower lobe), (lmsb, left main stem), (lup, left upper lobe), (lin, lingula)
        else: (med, mediastinum), (oth, other), (unk, unknown), (car, carina)
        """
        right_keys = ["locrhil", "locrlow", "locrmid", "locrmsb", "locrup"]
        left_keys = ["loclup", "loclmsb", "locllow", "loclhil", "loclin"]
        other_keys = ["loccar", "locmed", "locoth", "locunk"]

        right = any([pt_metadata[key][0] > 0 for key in right_keys])
        left = any([pt_metadata[key][0] > 0 for key in left_keys])
        other = any([pt_metadata[key][0] > 0 for key in other_keys])

        return np.array([int(right), int(left), int(other)])

    def order_slices(self, img_paths, slice_locations, reverse=False):
        sorted_ids = np.argsort(slice_locations)
        if reverse:
            sorted_ids = sorted_ids[::-1]
        sorted_img_paths = np.array(img_paths)[sorted_ids].tolist()
        sorted_slice_locs = np.sort(slice_locations).tolist()

        return sorted_img_paths, sorted_slice_locs

    def get_risk_factors(self, pt_metadata, screen_timepoint, return_dict=False):
        age_at_randomization = pt_metadata["age"][0]
        days_since_randomization = pt_metadata["scr_days{}".format(screen_timepoint)][0]
        current_age = age_at_randomization + days_since_randomization // 365

        age_start_smoking = pt_metadata["smokeage"][0]
        age_quit_smoking = pt_metadata["age_quit"][0]
        years_smoking = pt_metadata["smokeyr"][0]
        is_smoker = pt_metadata["cigsmok"][0]

        years_since_quit_smoking = 0 if is_smoker else current_age - age_quit_smoking

        education = (
            pt_metadata["educat"][0]
            if pt_metadata["educat"][0] != -1
            else pt_metadata["educat"][0]
        )

        race = pt_metadata["race"][0] if pt_metadata["race"][0] != -1 else 0
        race = 6 if pt_metadata["ethnic"][0] == 1 else race
        ethnicity = pt_metadata["ethnic"][0]

        weight = pt_metadata["weight"][0] if pt_metadata["weight"][0] != -1 else 0
        height = pt_metadata["height"][0] if pt_metadata["height"][0] != -1 else 0
        bmi = weight / (height**2) * 703 if height > 0 else 0  # inches, lbs

        prior_cancer_keys = [
            "cancblad",
            "cancbrea",
            "canccerv",
            "canccolo",
            "cancesop",
            "canckidn",
            "canclary",
            "canclung",
            "cancoral",
            "cancnasa",
            "cancpanc",
            "cancphar",
            "cancstom",
            "cancthyr",
            "canctran",
        ]
        cancer_hx = any([pt_metadata[key][0] == 1 for key in prior_cancer_keys])
        family_hx = any(
            [pt_metadata[key][0] == 1 for key in pt_metadata if key.startswith("fam")]
        )

        risk_factors = {
            "age": current_age,
            "race": race,
            "race_name": RACE_ID_KEYS.get(pt_metadata["race"][0], "UNK"),
            "ethnicity": ethnicity,
            "ethnicity_name": ETHNICITY_KEYS.get(ethnicity, "UNK"),
            "education": education,
            "bmi": bmi,
            "cancer_hx": cancer_hx,
            "family_lc_hx": family_hx,
            "copd": pt_metadata["diagcopd"][0],
            "is_smoker": is_smoker,
            "smoking_intensity": pt_metadata["smokeday"][0],
            "smoking_duration": pt_metadata["smokeyr"][0],
            "years_since_quit_smoking": years_since_quit_smoking,
            "weight": weight,
            "height": height,
            "gender": GENDER_KEYS.get(pt_metadata["gender"][0], "UNK"),
        }

        if return_dict:
            return risk_factors
        else:
            return np.array(
                [v for v in risk_factors.values() if not isinstance(v, str)]
            )

    def assign_splits(self, meta):
        if self.args.split_type == "institution_split":
            self.assign_institutions_splits(meta)
        elif self.args.split_type == "random":
            for idx in range(len(meta)):
                meta[idx]["split"] = np.random.choice(
                    ["train", "dev", "test"], p=self.args.split_probs
                )

    def assign_institutions_splits(self, meta):
        institutions = set([m["pt_metadata"]["cen"][0] for m in meta])
        institutions = sorted(institutions)
        institute_to_split = {
            cen: np.random.choice(["train", "dev", "test"], p=self.args.split_probs)
            for cen in institutions
        }
        for idx in range(len(meta)):
            meta[idx]["split"] = institute_to_split[meta[idx]["pt_metadata"]["cen"][0]]

    @property
    def CORRUPTED_PATHS(self):
        return pickle.load(open(CORRUPTED_PATHS, "rb"))

    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed NLST CT Cancer Risk {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
        class_balance = Counter([d["y"] for d in dataset])
        exams = set([d["exam"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group, len(dataset), len(exams), len(patients), class_balance
        )
        statement += "\n" + "Censor Times: {}".format(
            Counter([d["time_at_event"] for d in dataset])
        )
        statement
        return statement

    @property
    def GOOGLE_SPLITS(self):
        return pickle.load(open(GOOGLE_SPLITS_FILENAME, "rb"))

    def get_ct_annotations(self, sample):
        # correct empty lists of annotations
        if sample["series"] in self.annotations_metadata:
            self.annotations_metadata[sample["series"]] = {
                k: v
                for k, v in self.annotations_metadata[sample["series"]].items()
                if len(v) > 0
            }

        if sample["series"] in self.annotations_metadata:
            # store annotation(s) data (x,y,width,height) for each slice
            if (self.args.img_file_type == "dicom") and not sample["paths"][0].endswith(
                ".dcm"
            ):  # no file extension, so os.path.splitext breaks behavior
                sample["annotations"] = [
                    {
                        "image_annotations": self.annotations_metadata[
                            sample["series"]
                        ].get(os.path.basename(path), None)
                    }
                    for path in sample["paths"]
                ]
            else:  # expects file extension to exist, so use os.path.splitext
                sample["annotations"] = [
                    {
                        "image_annotations": self.annotations_metadata[
                            sample["series"]
                        ].get(os.path.splitext(os.path.basename(path))[0], None)
                    }
                    for path in sample["paths"]
                ]
        else:
            sample["annotations"] = [
                {"image_annotations": None} for path in sample["paths"]
            ]
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = copy.deepcopy(self.dataset[index])
        item = self.process_item(sample)
        if item is None:
            # new_index = random.randint(0, len(self.dataset) - 1)
            sample = copy.deepcopy(self.dataset[index + 1])
            item = self.process_item(sample)
        return item

    def process_item(self, sample):
        if self.args.use_annotations:
            sample = self.get_ct_annotations(sample)
        try:
            item = {}
            input_dict = self.get_images(sample["paths"], sample)

            x = input_dict["input"]

            if self.args.use_annotations:
                mask = torch.abs(input_dict["mask"])
                mask_area = mask.sum(dim=(-1, -2))
                item["volume_annotations"] = mask_area[0] / max(1, mask_area.sum())
                item["annotation_areas"] = mask_area[0] / (
                    mask.shape[-2] * mask.shape[-1]
                )
                mask_area = mask_area.unsqueeze(-1).unsqueeze(-1)
                mask_area[mask_area == 0] = 1
                item["image_annotations"] = mask / mask_area
                item["has_annotation"] = item["volume_annotations"].sum() > 0

            if self.args.use_risk_factors:
                item["risk_factors"] = sample["risk_factors"]

            item["x"] = x
            item["y"] = sample["y"]
            for key in CT_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

    def get_images(self, paths, sample):
        """
        Returns a stack of transformed images by their absolute paths.
        If cache is used - transformed images will be loaded if available,
        and saved to cache if not.
        """
        out_dict = {}
        if self.args.fix_seed_for_multi_image_augmentations:
            sample["seed"] = np.random.randint(0, 2**32 - 1)

        # get images for multi image input
        s = copy.deepcopy(sample)
        input_dicts = []
        for e, path in enumerate(paths):
            if self.args.use_annotations:
                s["annotations"] = sample["annotations"][e]
            input_dicts.append(self.input_loader.get_image(path, s))

        images = [i["input"] for i in input_dicts]
        input_arr = self.reshape_images(images)
        if self.args.use_annotations:
            masks = [i["mask"] for i in input_dicts]
            mask_arr = self.reshape_images(masks) if self.args.use_annotations else None

        # resample pixel spacing
        resample_now = (
            self.args.resample_pixel_spacing_prob > np.random.uniform()
        ) and self.args.resample_pixel_spacing
        if self.always_resample_pixel_spacing or resample_now:
            spacing = torch.tensor(sample["pixel_spacing"] + [1])
            input_arr = tio.ScalarImage(
                affine=torch.diag(spacing),
                tensor=input_arr.permute(0, 2, 3, 1),
            )
            input_arr = self.resample_transform(input_arr)
            if (input_arr.data.shape[-1] > self.args.num_images) or self.pad:
                input_arr = self.padding_transform(input_arr.data)
            else:
                input_arr = input_arr.data

            if self.args.use_annotations:
                mask_arr = tio.ScalarImage(
                    affine=torch.diag(spacing),
                    tensor=mask_arr.permute(0, 2, 3, 1),
                )
                mask_arr = self.resample_transform(mask_arr)
                if (mask_arr.data.shape[-1] > self.args.num_images) or self.pad:
                    mask_arr = self.padding_transform(mask_arr.data)
                else:
                    mask_arr = mask_arr.data

        elif self.args.resample_pixel_spacing:
            if (input_arr.data.shape[-1] > self.args.num_images) or self.pad:
                input_arr = self.padding_transform(input_arr.permute(0, 2, 3, 1))
                mask_arr = self.padding_transform(mask_arr.permute(0, 2, 3, 1))
            else:
                input_arr = input_arr.permute(0, 2, 3, 1)
                mask_arr = mask_arr.permute(0, 2, 3, 1)

        if self.args.resample_pixel_spacing:
            out_dict["input"] = input_arr.data.permute(0, 3, 1, 2)
            if self.args.use_annotations:
                out_dict["mask"] = mask_arr.data.permute(0, 3, 1, 2)
        else:
            out_dict["input"] = input_arr
            if self.args.use_annotations:
                out_dict["mask"] = mask_arr

        if out_dict["input"].shape[-1] != self.args.img_size[-1]:
            out_dict["input"] = F.interpolate(
                out_dict["input"],
                size=self.args.img_size,
                mode="bilinear",
            )
            if self.args.use_annotations:
                out_dict["mask"] = F.interpolate(
                    out_dict["mask"],
                    size=self.args.img_size,
                    mode="bilinear",
                )

        return out_dict

    def reshape_images(self, images):
        images = [im.unsqueeze(0) for im in images]
        images = torch.cat(images, dim=0)
        # Convert from (T, C, H, W) to (C, T, H, W)
        images = images.permute(1, 0, 2, 3)
        return images

    def get_slice_thickness_class(self, thickness):
        BINS = [1, 1.5, 2, 2.5]
        for i, tau in enumerate(BINS):
            if thickness <= tau:
                return i
        if self.args.slice_thickness_filter is not None:
            raise ValueError("THICKNESS > 2.5")
        return 4

class NLST_PidsWithAbnormalities(NLST_Survival_Dataset):
    def skip_sample(self, series_id, series_dict, exam_dict, pt_metadata):
        if 51 not in exam_dict["abnormalities"].get("sct_ab_desc", []):
            return True

        if not all(
            k in series_dict["slice_number"]
            for k in exam_dict["abnormalities"]["sct_slice_num"]
            if k != -1
        ):
            return True

        if len(series_dict["paths"]) <= 1:
            return True

        return False

    def create_dataset(self, split_group):
        dataset = []
        unique_slices = set()
        # add exams with abnormalities
        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            for exam_dict in exams:
                # check there is one series

                series = [
                    (series_id, series_dict)
                    for series_id, series_dict in exam_dict["image_series"].items()
                    if not self.is_localizer(series_dict["series_data"])
                ]
                series = [s for s in series if s[1]["slice_thickness"] is not None]

                if len(series) == 0:
                    continue

                for series_id, series_dict in series:
                    series_dict["series_data"] = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in series_dict["series_data"].items()
                    }

                    if self.skip_sample(series_id, series_dict, exam_dict, pt_metadata):
                        continue

                    series_data = series_dict["series_data"]
                    screen_timepoint = exam_dict["screen_timepoint"]
                    slice_numbers = series_dict["slice_number"]

                    pixel_spacing = series_dict["pixel_spacing"]
                    slice_thickness = series_dict["slice_thickness"]

                    img_paths = series_dict["paths"]

                    if img_paths[-1].endswith(".png"):
                        img_paths = [
                            (
                                PID2DICOM_DIRECTORY[pid]
                                + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                            ).replace(".png", ".dcm")
                            for path in img_paths
                        ]

                    slice_locations = series_dict["img_position"]
                    sorted_img_paths, sorted_slice_locs = self.order_slices(
                        img_paths, slice_locations
                    )
                    sorted_slice_numbers, _ = self.order_slices(
                        slice_numbers, slice_locations
                    )

                    days_since_rand = pt_metadata[
                        "scr_days{}".format(screen_timepoint)
                    ][0]
                    days_to_cancer_since_rand = pt_metadata["candx_days"][0]
                    days_to_cancer = days_to_cancer_since_rand - days_since_rand
                    has_future_cancer = int(days_to_cancer > -1)

                    for abnormality, abn_slice, abn_num in zip(
                        exam_dict["abnormalities"]["sct_ab_desc"],
                        exam_dict["abnormalities"]["sct_slice_num"],
                        exam_dict["abnormalities"]["sct_ab_num"],
                    ):
                        if abnormality != 51:
                            continue
                        path_index = slice_numbers.index(int(abn_slice))
                        path = img_paths[path_index]
                        if path in unique_slices:
                            continue
                        unique_slices.add(path)
                        exam_int = int(
                            "{}{}{}{}{}".format(
                                pid,
                                int(screen_timepoint),
                                int(series_id.split(".")[-1][-3:]),
                                int(abn_slice),
                                int(abn_num),
                            )
                        )
                        exam_intv2 = int(
                            "{}{}{}{}{}{}".format(
                                pid,
                                int(screen_timepoint),
                                series_id.split(".")[-1][:5],
                                series_id.split(".")[-1][-5:],
                                int(abn_slice),
                                int(abn_num),
                            )
                        )

                        dataset.append(
                            {
                                "path": path,
                                "slice_number": abn_slice,
                                "exam_str": "{}_{}".format(
                                    exam_dict["exam"], series_id
                                ),
                                "abn_num": abn_num,
                                "exam_3sid": exam_int,
                                "exam": exam_intv2,
                                "accession": exam_dict["accession_number"],
                                "series": series_id,
                                "study": series_data["studyuid"][0]
                                if "studyuid" in series_data
                                else exam_dict["accession_number"],
                                "screen_timepoint": screen_timepoint,
                                "pid": pid,
                                "y": has_future_cancer,
                                "slice_numbers": sorted_slice_numbers,
                                "slice_locations": sorted_slice_locs,
                                "paths": sorted_img_paths,
                                "pixel_spacing": pixel_spacing,
                                "slice_thickness": slice_thickness,
                                "has_abnormality": True,
                            }
                        )

        # add exams without abnormalities
        pids_with_abnormality = set(d["pid"] for d in dataset)
        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            # skip if pid has no exam annotated for an abnormality
            if pid not in pids_with_abnormality:
                continue

            for exam_dict in exams:
                # check there is one series
                series = [
                    (series_id, series_dict)
                    for series_id, series_dict in exam_dict["image_series"].items()
                    if not self.is_localizer(series_dict["series_data"])
                ]
                series = [s for s in series if s[1]["slice_thickness"] is not None]

                if len(series) == 0:
                    continue

                for series_id, series_dict in series:
                    series_dict["series_data"] = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in series_dict["series_data"].items()
                    }

                    if 51 in exam_dict["abnormalities"].get("sct_ab_desc", []):
                        continue

                    series_data = series_dict["series_data"]
                    screen_timepoint = exam_dict["screen_timepoint"]
                    slice_numbers = series_dict["slice_number"]

                    pixel_spacing = series_dict["pixel_spacing"]
                    slice_thickness = series_dict["slice_thickness"]

                    img_paths = series_dict["paths"]

                    if len(img_paths) <= 1:
                        continue

                    if img_paths[-1].endswith(".png"):
                        img_paths = [
                            (
                                PID2DICOM_DIRECTORY[pid]
                                + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                            ).replace(".png", ".dcm")
                            for path in img_paths
                        ]

                    slice_locations = series_dict["img_position"]
                    sorted_img_paths, sorted_slice_locs = self.order_slices(
                        img_paths, slice_locations
                    )
                    sorted_slice_numbers, _ = self.order_slices(
                        slice_numbers, slice_locations
                    )

                    days_since_rand = pt_metadata[
                        "scr_days{}".format(screen_timepoint)
                    ][0]
                    days_to_cancer_since_rand = pt_metadata["candx_days"][0]
                    days_to_cancer = days_to_cancer_since_rand - days_since_rand
                    has_future_cancer = int(days_to_cancer > -1)

                    exam_int = int(
                        "{}{}{}".format(
                            pid,
                            int(screen_timepoint),
                            int(series_id.split(".")[-1][-3:]),
                        )
                    )
                    exam_intv2 = int(
                        "{}{}{}{}".format(
                            pid,
                            int(screen_timepoint),
                            series_id.split(".")[-1][:5],
                            series_id.split(".")[-1][-5:],
                        )
                    )

                    dataset.append(
                        {
                            "path": sorted_img_paths[0],
                            "slice_number": -1,
                            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
                            "exam": exam_intv2,
                            "exam_3sid": exam_int,
                            "accession": exam_dict["accession_number"],
                            "series": series_id,
                            "study": series_data["studyuid"][0]
                            if "studyuid" in series_data
                            else exam_dict["accession_number"],
                            "screen_timepoint": screen_timepoint,
                            "pid": pid,
                            "y": has_future_cancer,
                            "slice_numbers": sorted_slice_numbers,
                            "slice_locations": sorted_slice_locs,
                            "paths": sorted_img_paths,
                            "pixel_spacing": pixel_spacing,
                            "slice_thickness": slice_thickness,
                            "has_abnormality": False,
                        }
                    )

        return dataset

    def process_item(self, sample):
        try:
            item = {}
            input_dict = self.input_loader.get_image(sample["path"], sample)
            x = input_dict["input"]
            item["x"] = x
            for key in CT_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            item["exam"] = str(item["exam"])

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed NLST CT Cancer Risk {} dataset with {} patients, {} exams with abnormalities, {} exams without abnormalities, {} nodules"
        exams = set([d["exam"] for d in dataset if d["has_abnormality"]])
        exams_without_abnormality = set(
            [d["exam"] for d in dataset if not d["has_abnormality"]]
        )
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group,
            len(patients),
            len(exams),
            len(exams_without_abnormality),
            len(dataset),
        )
        statement
        return statement

class NLST_Patches(NLST_Survival_Dataset):
    def __init__(self, args, split_group):
        super().__init__(args, split_group)
        self.augmentations = PatchAugmentations(args, split=split_group)
        self.pad = args.anatomix_pad_size
        self.anatomix_crop_size = args.anatomix_crop_size

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

        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row.get("split", "train"),
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            if (split != split_group) and not self.args.turn_off_splits:
                continue

            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():
                    if self.skip_sample(series_id, series_dict, exam_dict, pt_metadata):
                        continue

                    sample = self.get_volume_dict(
                        series_id, series_dict, exam_dict, pt_metadata, pid, split
                    )
                    if len(sample) == 0:
                        continue

                    if isinstance(sample, list):
                        dataset.extend(sample)
                    else:
                        dataset.append(sample)

        return dataset

    def get_volume_dict(
        self, series_id, series_dict, exam_dict, pt_metadata, pid, split
    ):
        series_data = series_dict["series_data"]
        img_paths = series_dict["paths"]
        slice_locations = series_dict["img_position"]
        pixel_spacing = series_dict["pixel_spacing"] + [series_dict["slice_thickness"]]
        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations, reverse=False
        )
        screen_timepoint = series_data["study_yr"][0]

        studyuid = exam_dict["exam"]

        if sorted_img_paths[-1].endswith(".png"):
            sorted_img_paths = [
                (
                    PID2DICOM_DIRECTORY[pid]
                    + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                ).replace(".png", ".dcm")
                for path in sorted_img_paths
            ]

        exam_int = int(
            "{}{}{}{}".format(
                pid,
                int(screen_timepoint),
                series_id.split(".")[-1][:5],
                series_id.split(".")[-1][-5:],
            )
        )

        samples = []
        for boxi, box in enumerate(series_dict["boxes"]):
            samples.append(
                {
                    "paths": sorted_img_paths,
                    "vol_exam": exam_int,
                    "exam": int(
                        "{}{}".format(
                            exam_int,
                            boxi + 1,
                        )
                    ),  # last 5 of study id + last 5 of series id
                    "study": studyuid,
                    "series": series_id,
                    "pid": pid,
                    "pixel_spacing": pixel_spacing,
                    "segmentation_path": series_dict["segmentation_path"],
                    "box": box,
                    "nifti_path": series_dict["nifti_path"],
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

        else:
            affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))
            label = np.load(sample["segmentation_path"])  # instance mask
            label = (label > 0) * 1.0  # binary mask
            image = load_nifti(sample["nifti_path"]).transpose(1, 2, 0)
            image = apply_windowing(image.astype(np.float64), -600, 1600)
            image = image // 256
            # slices = [
            #     self.input_loader.load_input(p, sample)["input"]
            #     for p in sample["paths"]
            # ]
            # image = np.stack(slices, axis=-1)
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
                "dataset": "nlst",
            }
        else:
            h1, w1, _ = self.anatomix_crop_size
            d1 = self.pad[-1]

            y = 1
            if self.split_group in ["train", "dev"]:
                lung_mask_path = (
                    "/path/to/local/cache/lung_ct/nlst_lung_mask/sample_{}.npy".format(
                        sample["vol_exam"]
                    )
                )
                if (
                    np.random.uniform(0, 1) < self.args.sample_negative_ratio
                ) and os.path.exists(lung_mask_path):
                    # Generate a negative box of the same size as the original box, but at a different location

                    # Load lung mask (assume path is in sample["lung_mask_path"])
                    lung_mask = np.load(lung_mask_path)  # binary mask, shape (D, H, W)
                    if (img_h, img_w) != tuple(img_size):
                        lung_mask = torch.tensor(lung_mask) * 1.0
                        lung_mask = lung_mask.unsqueeze(1)
                        lung_mask = F.interpolate(
                            lung_mask, size=(img_size[0], img_size[1]), mode="nearest"
                        )
                        lung_mask = (
                            lung_mask.squeeze(1).numpy().transpose(1, 2, 0)
                        )  # (H, W, D)
                    else:
                        lung_mask = lung_mask.transpose(1, 2, 0)  # (H, W, D)

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

            if getattr(self.args, "augment_before_cropping", False):
                # cropping should be done after augmentation on the full image
                sample = self.augmentations(
                    {
                        "image": image,
                        "label": label,
                    }
                )
                cbbox = random_pad_3d_box(
                    box,
                    image,
                    min_height=h1,
                    min_width=w1,
                    min_depth=d1,
                    random_hw=False,
                )
                sample["image"] = image[cbbox]
                sample["label"] = label[cbbox]
                sample = self.augmentations.crop_pad_transform(sample)

            else:
                cbbox = random_pad_3d_box(
                    box,
                    image,
                    min_height=h1,
                    min_width=w1,
                    min_depth=d1,
                    random_hw=False,
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
                "dataset": "nlst",
            }
        return item

    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed NLST CT Cancer Risk {} dataset with {} records, {} exams, {} patients\n"
        exams = set([d["exam"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group,
            len(dataset),
            len(exams),
            len(patients),
        )
        return statement

class NLST_Confidence_Flat(NLST_Patches):
    def get_volume_dict(
        self, series_id, series_dict, exam_dict, pt_metadata, pid, split
    ):
        series_data = series_dict["series_data"]
        img_paths = series_dict["paths"]
        slice_locations = series_dict["img_position"]
        pixel_spacing = series_dict["pixel_spacing"] + [series_dict["slice_thickness"]]
        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations, reverse=False
        )
        screen_timepoint = series_data["study_yr"][0]

        studyuid = exam_dict["exam"]

        if sorted_img_paths[-1].endswith(".png"):
            sorted_img_paths = [
                (
                    PID2DICOM_DIRECTORY[pid]
                    + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                ).replace(".png", ".dcm")
                for path in sorted_img_paths
            ]

        exam_int = int(
            "{}{}{}{}".format(
                pid,
                int(screen_timepoint),
                series_id.split(".")[-1][:5],
                series_id.split(".")[-1][-5:],
            )
        )

        samples = []
        for boxi, box in enumerate(series_dict["boxes"]):
            samples.append(
                {
                    "paths": sorted_img_paths,
                    "vol_exam": exam_int,
                    "exam": int(
                        "{}{}".format(
                            exam_int,
                            boxi + 1,
                        )
                    ),  # last 5 of study id + last 5 of series id
                    "study": studyuid,
                    "series": series_id,
                    "pid": pid,
                    "pixel_spacing": pixel_spacing,
                    "segmentation_path": series_dict["segmentation_path"],
                    "box": box,
                    "nifti_path": series_dict["nifti_path"],
                    "is_true_box": True,
                }
            )

        boxi_offset = len(samples)
        predicted_boxes = series_dict.get("predicted_boxes", [])
        if predicted_boxes is None:
            predicted_boxes = []

        for boxi, (box, voxels) in enumerate(predicted_boxes):
            if voxels * np.prod(pixel_spacing) < self.args.min_nodule_volume:
                continue
            samples.append(
                {
                    "paths": sorted_img_paths,
                    "vol_exam": exam_int,
                    "exam": int(
                        "{}{}".format(exam_int, boxi + 1 + boxi_offset)
                    ),  # last 5 of study id + last 5 of series id
                    "study": studyuid,
                    "series": series_id,
                    "pid": pid,
                    "pixel_spacing": pixel_spacing,
                    "segmentation_path": series_dict["segmentation_path"],
                    "box": box,
                    "nifti_path": series_dict["nifti_path"],
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
            label = np.load(sample["segmentation_path"])  # instance mask
            label = (label > 0) * 1.0  # binary mask
            image = load_nifti(sample["nifti_path"]).transpose(1, 2, 0)
            image = apply_windowing(image.astype(np.float64), -600, 1600)
            image = image // 256
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
                "dataset": "nlst",
            }
        else:
            h1, w1, _ = self.anatomix_crop_size
            d1 = self.pad[-1]

            y = 1
            if self.split_group in ["train", "dev"]:
                if not is_true_box:
                    segmentation = pickle.load(
                        open(
                            f"/path/to/local/cache/lung_ct/luna-stmix/last/sample_{sample['vol_exam']}.hiddens",
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
            x = torch.cat([sample["image"], sample["label"]], dim=0)

            item = {
                "x": x,
                "y": y,
                "exam": sample["exam"],
                "pid": sample["pid"],
                "vol_exam": sample["vol_exam"],
                "dataset": "nlst",
            }
        return item

class NLST_PidsWithAbnormalitiesForNnUNet(NLST_PidsWithAbnormalities):
    def create_dataset(self, split_group):
        series_to_use = pickle.load(
            open(
                "/data/rbg/users/pgmikhael/current/Sybil/notebooks/pid2series.p",
                "rb",
            )
        )
       
        series_to_use = set(s[1] for _, series in series_to_use.items() for s in series)
        dataset_ = super().create_dataset(split_group)
        dataset = []
        vol_exams = set()
        for d in dataset_:
            if d["exam"] not in series_to_use:
                continue
            # if not d["has_abnormality"]:
            #     continue
            
            if str(d["exam"])[:17] in vol_exams:
                continue

            vol_exams.add(str(d["exam"])[:17])

            dataset.append(d)

        return dataset

    def process_item(self, sample):
        try:
            path = "/path/to/local/cache/lung_ct/nlst_nifti/sample_{}.nii.gz".format(
                sample["exam"]
            )

            if not os.path.exists(path):
                image = pydicom_to_nifti(sample["paths"], path).transpose(
                    1, 2, 0
                )  # (Y, X, Z)
            else:
                try:
                    image = load_nifti(path).transpose(1, 2, 0)  # (Y, X, Z)
                except:
                    image = pydicom_to_nifti(sample["paths"], path).transpose(
                        1, 2, 0
                    )  # (Y, X, Z)

            affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))
            image = apply_windowing(image.astype(np.float64), -600, 1600)
            image = image // 256
            image = MetaTensor(
                image,
                affine=affine,
                dtype=torch.float32,
            )

            # Ensure image and label have the correct spatial size (args.img_size)
            img_size = self.args.img_size  # (H, W)
            img_h, img_w = image.shape[0], image.shape[1]
            if (img_h, img_w) != tuple(img_size):
                # image: (H, W, D), label: (H, W, D)
                # Add batch and channel dims for interpolation: (1, 1, D, H, W)
                image_ = image.permute(2, 0, 1).unsqueeze(1)

                image_ = F.interpolate(
                    image_,
                    size=(img_size[0], img_size[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                # Remove batch/channel dims and permute back: (H, W, D)
                image = image_.squeeze(1)

            # crop to slice of interest
            # slice_id = sample["paths"].index(sample["path"])
            # image = image[max(0, slice_id - 16) : min(image.shape[0], slice_id + 16)]

            item = {
                "x": image[None],
                "exam": str(sample["exam"]),
                "pid": sample["pid"],
                "vol_exam": str(sample["exam"]),
                "y": 1,  # assuming all samples are positive
            }

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

class NLST_LUNA25_Annotated_Cancers_ForNnUNet(NLST_PidsWithAbnormalitiesForNnUNet):
    def create_dataset(self, split_group):
        self.nlst_luna25 = pd.read_csv(
            "/data/rbg/shared/datasets/NLST/NLST/LUNA25_Public_Training_Development_Data.csv"
        )
        nlst_luna25_series = list(self.nlst_luna25["SeriesInstanceUID"])
        annotated_cancers = list(self.annotations_metadata.keys())
        series_to_use = set(nlst_luna25_series + annotated_cancers)

        dataset = []

        # add exams with abnormalities
        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():
                    series_dict["series_data"] = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in series_dict["series_data"].items()
                    }
                    if series_id not in series_to_use:
                        continue

                    series_data = series_dict["series_data"]
                    screen_timepoint = exam_dict["screen_timepoint"]

                    exam_intv2 = "{}{}{}{}".format(
                        pid,
                        int(screen_timepoint),
                        series_id.split(".")[-1][:5],
                        series_id.split(".")[-1][-5:],
                    )

                    pixel_spacing = series_dict["pixel_spacing"]
                    slice_thickness = series_dict["slice_thickness"]

                    img_paths = series_dict["paths"]

                    if img_paths[-1].endswith(".png"):
                        img_paths = [
                            (
                                PID2DICOM_DIRECTORY[pid]
                                + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                            ).replace(".png", ".dcm")
                            for path in img_paths
                        ]

                    slice_locations = series_dict["img_position"]
                    sorted_img_paths, sorted_slice_locs = self.order_slices(
                        img_paths, slice_locations
                    )

                    days_since_rand = pt_metadata[
                        "scr_days{}".format(screen_timepoint)
                    ][0]
                    days_to_cancer_since_rand = pt_metadata["candx_days"][0]
                    days_to_cancer = days_to_cancer_since_rand - days_since_rand
                    has_future_cancer = int(days_to_cancer > -1)

                    dataset.append(
                        {
                            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
                            "exam": exam_intv2,
                            "accession": exam_dict["accession_number"],
                            "series": series_id,
                            "study": series_data["studyuid"][0]
                            if "studyuid" in series_data
                            else exam_dict["accession_number"],
                            "screen_timepoint": screen_timepoint,
                            "pid": pid,
                            "y": has_future_cancer,
                            "slice_locations": sorted_slice_locs,
                            "paths": sorted_img_paths,
                            "pixel_spacing": pixel_spacing,
                            "slice_thickness": slice_thickness,
                            "has_abnormality": True,
                        }
                    )

        return dataset

class NLST_TestSet_ForNnUNet(NLST_PidsWithAbnormalitiesForNnUNet):
    def create_dataset(self, split_group):
        test_dataset = pickle.load(
            open(
                "/data/rbg/users/pgmikhael/current/notebooks/LungCT/Sybil/ResultsDir/nlst_dataset.p",
                "rb",
            )
        )["test"]
        test_set_exams = set(str(sample["exam"]) for sample in test_dataset)
        dataset = []

        # add exams with abnormalities
        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():
                    series_dict["series_data"] = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in series_dict["series_data"].items()
                    }
                    exam3s = "{}{}{}".format(
                        pt_metadata["pid"][0],
                        exam_dict["screen_timepoint"],
                        int(series_id.split(".")[-1][-3:]),
                    )
                    if exam3s not in test_set_exams:
                        continue

                    series_data = series_dict["series_data"]
                    screen_timepoint = exam_dict["screen_timepoint"]

                    exam_intv2 = "{}{}{}{}".format(
                        pid,
                        int(screen_timepoint),
                        series_id.split(".")[-1][:5],
                        series_id.split(".")[-1][-5:],
                    )

                    pixel_spacing = series_dict["pixel_spacing"]
                    slice_thickness = series_dict["slice_thickness"]

                    img_paths = series_dict["paths"]

                    if img_paths[-1].endswith(".png"):
                        img_paths = [
                            (
                                PID2DICOM_DIRECTORY[pid]
                                + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                            ).replace(".png", ".dcm")
                            for path in img_paths
                        ]

                    slice_locations = series_dict["img_position"]
                    sorted_img_paths, sorted_slice_locs = self.order_slices(
                        img_paths, slice_locations
                    )

                    days_since_rand = pt_metadata[
                        "scr_days{}".format(screen_timepoint)
                    ][0]
                    days_to_cancer_since_rand = pt_metadata["candx_days"][0]
                    days_to_cancer = days_to_cancer_since_rand - days_since_rand
                    has_future_cancer = int(days_to_cancer > -1)

                    dataset.append(
                        {
                            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
                            "exam": exam_intv2,
                            "accession": exam_dict["accession_number"],
                            "series": series_id,
                            "study": series_data["studyuid"][0]
                            if "studyuid" in series_data
                            else exam_dict["accession_number"],
                            "screen_timepoint": screen_timepoint,
                            "pid": pid,
                            "y": has_future_cancer,
                            "slice_locations": sorted_slice_locs,
                            "paths": sorted_img_paths,
                            "pixel_spacing": pixel_spacing,
                            "slice_thickness": slice_thickness,
                            "has_abnormality": True,
                        }
                    )

        return dataset

class NLST_SegmentationProcessing(NLST_Survival_Dataset):
    def __init__(self, args, split_group):
        super().__init__(args, split_group)
        self.augmentations = PatchAugmentations(args, split=split_group)
        self.pad = args.anatomix_pad_size
        self.anatomix_crop_size = args.anatomix_crop_size

    def create_dataset(self, split_group):
        nlst_luna25 = pd.read_csv(
            "/data/rbg/shared/datasets/NLST/NLST/LUNA25_Public_Training_Development_Data.csv"
        ).to_dict(orient="records")
        nlst_luna25_records = defaultdict(list)
        for row in nlst_luna25:
            nlst_luna25_records[
                "{}_{}".format(row["PatientID"], row["SeriesInstanceUID"])
            ].append(row)

        dataset = []
        segmentation_files = {}
        for segdir in [
            "luna-stmix/last",
            "nlst_abnormalities51_nnunet_segmentation_abn/last",
        ]:
            for segfile in os.listdir(
                os.path.join("/path/to/local/cache/lung_ct", segdir)
            ):
                exam = segfile.split("_")[1].split(".")[0]
                exam = exam[:17] if len(exam) > 17 else exam
                segmentation_files[exam] = os.path.join(
                    "/path/to/local/cache/lung_ct", segdir, segfile
                )
        nifti_files = {}
        for nifti_file in os.listdir("/path/to/local/cache/lung_ct/nlst_nifti/"):
            exam = nifti_file.split("_")[1].split(".")[0]
            exam = exam[:17] if len(exam) > 17 else exam
            nifti_files[exam] = os.path.join(
                "/path/to/local/cache/lung_ct/nlst_nifti/", nifti_file
            )

        # add exams with abnormalities
        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, exams, pt_metadata, split = (
                mrn_row["pid"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
                mrn_row.get("split", "train"),
            )

            if (split != split_group) and not self.args.turn_off_splits:
                continue

            for exam_dict in exams:
                # check there is one series

                for series_id, series_dict in exam_dict["image_series"].items():
                    series_dict["series_data"] = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in series_dict["series_data"].items()
                    }

                    if self.skip_sample(series_id, series_dict, exam_dict, pt_metadata):
                        continue

                    screen_timepoint = exam_dict["screen_timepoint"]
                    examid = "{}{}{}{}".format(
                        pid,
                        int(screen_timepoint),
                        series_id.split(".")[-1][:5],
                        series_id.split(".")[-1][-5:],
                    )
                    if examid not in segmentation_files:
                        continue

                    series_data = series_dict["series_data"]
                    slice_numbers = series_dict["slice_number"]
                    pixel_spacing = series_dict["pixel_spacing"]
                    slice_thickness = series_dict["slice_thickness"]
                    img_paths = series_dict["paths"]
                    if img_paths[-1].endswith(".png"):
                        img_paths = [
                            (
                                PID2DICOM_DIRECTORY[pid]
                                + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                            ).replace(".png", ".dcm")
                            for path in img_paths
                        ]

                    slice_locations = series_dict["img_position"]
                    sorted_img_paths, sorted_slice_locs = self.order_slices(
                        img_paths, slice_locations
                    )
                    sorted_slice_numbers, _ = self.order_slices(
                        slice_numbers, slice_locations
                    )

                    # get annotations if they exist
                    if series_id in self.annotations_metadata:
                        cancer_annotations = self.annotations_metadata[series_id]
                    else:
                        cancer_annotations = {}
                    if "{}_{}".format(pid, series_id) in nlst_luna25_records:
                        # contains coordx, coordy, coordz, label
                        luna25_annotations = nlst_luna25_records[
                            "{}_{}".format(pid, series_id)
                        ]
                    else:
                        luna25_annotations = []

                    dataset.append(
                        {
                            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
                            "exam": examid,
                            "accession": exam_dict["accession_number"],
                            "series": series_id,
                            "study": series_data["studyuid"][0]
                            if "studyuid" in series_data
                            else exam_dict["accession_number"],
                            "screen_timepoint": screen_timepoint,
                            "pid": pid,
                            "slice_numbers": sorted_slice_numbers,
                            "slice_locations": sorted_slice_locs,
                            "paths": sorted_img_paths,
                            "pixel_spacing": pixel_spacing,
                            "slice_thickness": slice_thickness,
                            "segmentation_path": segmentation_files[examid],
                            "cancer_annotations": cancer_annotations,
                            "luna25_annotations": luna25_annotations,
                            "nifti_path": nifti_files.get(examid, None),
                        }
                    )

        return dataset

    def process_item(self, sample):
        try:
            # load the nifti if it exists, otherwise convert from dicom
            path = sample["nifti_path"]
            if path is not None:
                nifti_img = sitk.ReadImage(path)
                image = sitk.GetArrayFromImage(nifti_img)  # z, y, x
                image = image.transpose(1, 2, 0)  # y, x, z
            try:
                image, nifti_img = pydicom_to_nifti(
                    sample["paths"], path, return_nifti=True
                )  # y, x, z
            except:
                return

            # load the segmentation
            segmentation = torch.load(sample["segmentation_path"])
            sparse_segmentation = segmentation["sparse_segmentation"]
            nodule_volumes = segmentation["nodule_volumes"]
            nodule_cancer_labels = segmentation["nodule_cancer_labels"]
            nodule_luna25_labels = segmentation["nodule_luna25_labels"]
            valid_nodule_ids = segmentation["nodule_ids"]
            segmentation = sparse_segmentation.to_dense()

            labels_out = cc3d.connected_components(segmentation, return_N=False)
            bboxes = cc3d.statistics(labels_out)["bounding_boxes"][
                1:
            ]  # skip background
            # convert into list of dicts with keys y_start, y_stop, x_start, x_stop, z_start, z_stop
            bboxes = [
                {
                    "y_start": b[0].start,
                    "y_stop": b[0].stop,
                    "x_start": b[1].start,
                    "x_stop": b[1].stop,
                    "z_start": b[2].start,
                    "z_stop": b[2].stop,
                }
                for b in bboxes
            ]

            affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))
            image = apply_windowing(image.astype(np.float64), -600, 1600)
            image = image // 256
            image = MetaTensor(
                image,
                affine=affine,
                dtype=torch.float32,
            )
            segmentation = MetaTensor(
                segmentation,
                affine=affine,
                dtype=torch.float32,
            )

            img_size = self.args.img_size
            img_h, img_w = image.shape[0], image.shape[1]
            if (img_h, img_w) != tuple(img_size):
                # image: (H, W, D), label: (H, W, D)
                # Add batch and channel dims for interpolation: (1, 1, D, H, W)
                image_ = image.permute(2, 0, 1).unsqueeze(1)
                image_ = F.interpolate(
                    image_,
                    size=(img_size[0], img_size[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                # Remove batch/channel dims and permute back: (H, W, D)
                image = image_.squeeze(1).permute(1, 2, 0)

            # make patches around each component
            patches = []
            patch_h, patch_w, patch_z = (128, 128, 10)  # hard code
            for box in bboxes:
                cbbox = random_pad_3d_box(
                    box,
                    image,
                    min_height=patch_h,
                    min_width=patch_w,
                    min_depth=patch_z,
                    random_hw=False,
                    random_d=False,
                )
                patch = {
                    "image": image[cbbox],
                    "label": segmentation[cbbox],
                }
                patch = self.augmentations.test_transforms(patch)
                patch = torch.cat(
                    [
                        patch["image"].permute(0, 3, 1, 2),
                        patch["label"].permute(0, 3, 1, 2),
                    ],
                    dim=0,
                )
                patches.append(patch)

            patches = torch.stack(patches)

            item = {
                "x": patches,
                "exam": str(sample["exam"]),
                "segmentation": segmentation,
                "nodule_volumes": nodule_volumes,
                "nodule_cancer_labels": nodule_cancer_labels,
                "nodule_luna25_labels": nodule_luna25_labels,
                "nodule_ids": valid_nodule_ids,
                "pid": sample["pid"],
            }

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

class NLST_Survival_Dataset2(NLST_Survival_Dataset):
    def __init__(self, args, split_group):
        """
        NLST Dataset
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(NLST_Survival_Dataset, self).__init__()

        self.split_group = split_group
        self.args = args
        self.augmentations = (
            PillarAugmentations(args, split=split_group)
            if args.use_pillar_processing
            else FullAugmentations(args, split=split_group)
        )
        patch_args = copy.deepcopy(args)
        patch_args.anatomix_crop_size = [128, 128, 32]
        patch_args.anatomix_pad_size = [15, 15, 32]
        self.patch_augmentations = PatchAugmentations(patch_args, split=split_group)

        try:
            self.metadata_json = json.load(open(args.dataset_file_path, "r"))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

        if self.args.region_annotations_filepath:
            self.annotations_metadata = json.load(
                open(self.args.region_annotations_filepath, "r")
            )
            self.annotations_metadata["exams"] = set(self.annotations_metadata["exams"])
            self.annotations_metadata["lung_masks"] = set(
                self.annotations_metadata["lung_masks"]
            )
            self.annotations_metadata["pillar_exams"] = set(
                self.annotations_metadata["pillar_exams"]
            )
            self.nlst_luna25 = pickle.load(
                open(
                    "/data/rbg/users/pgmikhael/current/Sybil/notebooks/NoduleGrowth/luna25_nlst_segmentation_dataset.p",
                    "rb",
                )
            )
        else:
            self.annotations_metadata = {}

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        print(self.get_summary_statement(self.dataset, split_group))

        if args.class_bal:
            label_dist = [d[args.class_bal_key] for d in self.dataset]
            label_counts = Counter(label_dist)
            weight_per_label = 1.0 / len(label_counts)
            label_weights = {
                label: weight_per_label / count for label, count in label_counts.items()
            }

            print("Class counts are: {}".format(label_counts))
            print("Label weights are {}".format(label_weights))
            self.weights = [label_weights[d[args.class_bal_key]] for d in self.dataset]
        self.metadata_json = None

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
        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row.get("split", "train"),
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            if (split != split_group) and not self.args.turn_off_splits:
                continue

            valid_series = defaultdict(list)
            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():
                    series_dict["series_data"] = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in series_dict["series_data"].items()
                    }
                    if self.skip_sample(series_id, series_dict, exam_dict, pt_metadata):
                        continue

                    sample = self.get_volume_dict(
                        series_id,
                        series_dict,
                        exam_dict,
                        pt_metadata,
                        pid,
                        split,
                    )
                    if len(sample) == 0:
                        continue

                    valid_series[(pid, sample["screen_timepoint"])].append(sample)

            for key, valid_series_ in valid_series.items():
                if len(valid_series_) == 0:
                    continue
                else:
                    numslices = [s["num_original_slices"] for s in valid_series_]
                    max_slices_idx = numslices.index(max(numslices))
                    sample = valid_series_[max_slices_idx]

                if isinstance(sample, list):
                    dataset.extend(sample)
                else:
                    dataset.append(sample)

        return dataset

    def skip_sample(self, series_id, series_dict, exam_dict, pt_metadata):
        if super().skip_sample(series_id, series_dict, exam_dict, pt_metadata):
            return True

        pid = pt_metadata["pid"][0]

        if self.split_group == "test":
            return False

        # skip if no lung mask
        examid = get_examid(pid, exam_dict["screen_timepoint"], series_id)

        # if examid not in self.annotations_metadata["lung_masks"]:
        #     return True

        # skip if not annotated and other annotated exam exists for patient
        all_exams = [
            get_examid(pid, exam_dict["screen_timepoint"], sid)
            for sid, sdict in exam_dict["image_series"].items()
            if not self.is_localizer(sdict["series_data"])
        ]
        all_luna_exams = [
            f"{pt_metadata['pid'][0]}_{sid}"
            for sid, sdict in exam_dict["image_series"].items()
            if not self.is_localizer(sdict["series_data"])
        ]

        annotated_exams = [
            examid
            for examid in all_exams
            if examid in self.annotations_metadata["exams"]
        ]
        luna25_exams = [
            f"{pid}_{sid}"
            for sid in exam_dict["image_series"].keys()
            if f"{pid}_{sid}" in self.nlst_luna25
        ]

        if len(luna25_exams) > 0 and (f"{pid}_{series_id}" not in self.nlst_luna25):
            return True

        elif (
            (len(luna25_exams) == 0)
            and (len(annotated_exams) > 0)
            and (examid not in annotated_exams)
        ):
            return True

        # otherwise skip if slice thickness is not the thinnest in exam
        if series_dict["slice_thickness"] is None:
            return True

        slice_thicknesses = [
            d["slice_thickness"]
            for s, d in exam_dict["image_series"].items()
            if not self.is_localizer(d["series_data"])
        ]
        if len(slice_thicknesses) == 0:
            return True

        if (examid not in annotated_exams) and (
            f"{pid}_{series_id}" not in self.nlst_luna25
        ):
            min_slice_thickness = min([s for s in slice_thicknesses if s is not None])
        elif f"{pid}_{series_id}" in self.nlst_luna25:
            slice_thicknesses = [
                s
                for s, e in zip(slice_thicknesses, all_luna_exams)
                if (e in self.nlst_luna25) and (s is not None)
            ]
            min_slice_thickness = min(slice_thicknesses)
        elif examid in annotated_exams:
            slice_thicknesses = [
                s
                for s, e in zip(slice_thicknesses, all_exams)
                if (e in annotated_exams) and (s is not None)
            ]
            min_slice_thickness = min(slice_thicknesses)
        if series_dict["slice_thickness"] > min_slice_thickness:
            return True

        return False

    def get_volume_dict(
        self,
        series_id,
        series_dict,
        exam_dict,
        pt_metadata,
        pid,
        split,
    ):
        img_paths = series_dict["paths"]
        slice_locations = series_dict["img_position"]
        slice_numbers = series_dict["slice_number"]
        series_data = series_dict["series_data"]
        device = DEVICE_ID[DEVICE_TO_NAME[series_data["manufacturer"][0]]]
        screen_timepoint = series_data["study_yr"][0]
        assert screen_timepoint == exam_dict["screen_timepoint"]

        if img_paths[-1].endswith(".png"):
            img_paths = [
                (
                    PID2DICOM_DIRECTORY[pid]
                    + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                ).replace(".png", ".dcm")
                for path in img_paths
            ]

        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations, reverse=self.args.reverse_slice_order
        )
        sorted_slice_numbers, _ = self.order_slices(slice_numbers, slice_locations)

        # if not sorted_img_paths[0].startswith(self.args.img_dir):
        #     sorted_img_paths = [
        #         self.args.img_dir
        #         + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
        #         for path in sorted_img_paths
        #     ]

        # if sorted_img_paths[0].endswith(".dcm.png"):
        #     sorted_img_paths = [p.replace(".dcm.png", ".png") for p in sorted_img_paths]
        y, y_seq, y_mask, time_at_event = self.get_label(pt_metadata, screen_timepoint)

        examid = "{}{}{}{}".format(
            pid,
            int(screen_timepoint),
            series_id.split(".")[-1][:5],
            series_id.split(".")[-1][-5:],
        )
        segpath = None
        if examid in self.annotations_metadata["exams"]:
            segpath = f"/path/to/local/cache/lung_ct/nlst_abnormalities51_nnunet_sparse_segmentation/sample_{examid}.pt"

        sample = {
            "split": split,
            "paths": sorted_img_paths,
            "slice_locations": sorted_slice_locs,
            "slice_numbers": sorted_slice_numbers,
            "y": int(y),
            "time_at_event": time_at_event,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
            "exam": examid,
            "accession": exam_dict["accession_number"],
            "series": series_id,
            "screen_timepoint": screen_timepoint,
            "pid": pid,
            "device": device,
            "institution": pt_metadata["cen"][0],
            "cancer_laterality": self.get_cancer_side(pt_metadata),
            "num_original_slices": len(series_dict["paths"]),
            "pixel_spacing": series_dict["pixel_spacing"]
            + [series_dict["slice_thickness"]],
            "slice_thickness": series_dict["slice_thickness"],
            "segmentation_path": segpath,
            "has_annotation": int(segpath is not None),
            "has_annotation_and_future_cancer": int(
                (segpath is not None) and (y == 1),
            ),
            "lung_mask_path": f"/path/to/local/cache/lung_ct/nlst_lung_mask/sample_{examid}.npy",
            "confidence_path": self.annotations_metadata["confidence_files"].get(
                examid, None
            ),
        }

        if self.args.use_risk_factors:
            sample["risk_factors"] = self.get_risk_factors(
                pt_metadata, screen_timepoint, return_dict=False
            )

        return sample

    def load_image(self, sample):
        cache_file = (
            os.path.join(self.args.cache_path, f"nlst_sample_{sample['exam']}.pt")
            if self.args.cache_path
            else None
        )
        if (
            (self.split_group == "dev")
            and self.args.cache_path
            and os.path.exists(cache_file)
        ):
            image = torch.load(cache_file, weights_only=False)
        else:
            affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))
            # load the nifti if it exists, otherwise convert from dicom
            path = self.annotations_metadata["nifti_files"].get(sample["exam"], None)
            if path is not None:
                nifti_img = sitk.ReadImage(path)
                image = sitk.GetArrayFromImage(nifti_img)  # z, y, x
                image = image.transpose(1, 2, 0)  # y, x, z
            else:
                try:
                    path = f"/path/to/local/cache/lung_ct/nlst_nifti/sample_{sample['exam']}.nii.gz"
                    image = pydicom_to_nifti(
                        sample["paths"], path, save_nifti=True
                    )  # y, x, z
                    self.annotations_metadata["nifti_files"][sample["exam"]] = path
                except:
                    return

        if self.args.cache_path and (self.split_group == "dev"):
            torch.save(image, cache_file)

        image = MetaTensor(
            image,
            affine=affine,
            dtype=torch.float32,
        )

        return image

    def process_image_for_pillar(self, image):
        # min-max normalize
        minmax = (image - image.min()) / (image.max() - image.min() + 1e-8)
        minmax = torch.clamp(minmax, 0, 1)
        # apply anatomical windows
        image = [
            apply_pillar_windowing(image, window["center"], window["width"])
            for bodypart, window in ANATOMICAL_WINDOWS["CT"].items()
        ] + [minmax]
        image = torch.concat(image, 0)  #  channels, y, x, z
        return image

    def process_image_default(self, image):
        image = apply_windowing(image, -600, 1600)
        image = image.float() // 256
        return image

    def zoom_image(self, image, target_size):
        resized = apply_windowing(image.astype(np.float64), -600, 1600)
        resized = torch.tensor(resized, dtype=torch.float32) // 256
        resized = resized.permute(2, 0, 1).unsqueeze(1)
        resized = F.interpolate(
            resized,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        resized = resized.squeeze(1).permute(1, 2, 0)
        return resized

    def load_segmentation(self, sample):
        pid_series_key = "{}_{}".format(sample["pid"], sample["series"])
        if pid_series_key in self.nlst_luna25:
            segmentation = torch.load(
                f"/path/to/local/cache/lung_ct/nlst_luna25_sparse_segmentation/sample_{sample['exam']}.pt"
            )
            sparse_segmentation = segmentation["sparse_segmentation"]
            nodule_volumes = segmentation["nodule_volumes"]
            nodule_ids = segmentation["nodule_ids"]
            segmentation_ = sparse_segmentation.to_dense()

            nodule_luna25_labels_dict = {}
            for nodule_dict in self.nlst_luna25[pid_series_key]:
                y, x, z = (
                    int(nodule_dict["ycenter"]),
                    int(nodule_dict["xcenter"]),
                    int(nodule_dict["zcenter"]),
                )
                nodid = int(segmentation_[y, x, z].item())
                nodule_luna25_labels_dict[nodid] = nodule_dict["nodule_label"]

            nodule_luna25_labels = torch.tensor(
                [nodule_luna25_labels_dict.get(nid.item(), 0) for nid in nodule_ids]
            )
            nodule_has_luna25_labels = torch.tensor(
                [nid.item() in nodule_luna25_labels_dict for nid in nodule_ids]
            )
            nodule_cancer_labels = torch.zeros(len(nodule_ids), dtype=torch.long)
            nodule_has_cancer_labels = torch.zeros(len(nodule_ids), dtype=torch.long)

            segmentation_ = segmentation_.permute(2, 0, 1).unsqueeze(1)
            segmentation_ = F.interpolate(
                segmentation_,
                size=(self.args.img_size[0], self.args.img_size[1]),
                mode="nearest-exact",
            )
            # Remove batch/channel dims and permute back: (H, W, D)
            segmentation_ = segmentation_.squeeze(1).permute(1, 2, 0)
            num_nodules = len(nodule_ids)

        elif sample["segmentation_path"] is not None:
            segmentation = torch.load(sample["segmentation_path"])
            sparse_segmentation = segmentation["sparse_segmentation"]
            nodule_volumes = segmentation["nodule_volumes"]
            nodule_ids = segmentation["nodule_ids"]
            if "nodule_cancer_labels" in segmentation:
                nodule_cancer_labels = segmentation["nodule_cancer_labels"]
                nodule_luna25_labels = segmentation["nodule_luna25_labels"]
                nodule_has_cancer_labels = segmentation["nodule_has_cancer_labels"]
                nodule_has_luna25_labels = segmentation["nodule_has_luna25_labels"]
            else:
                nodule_cancer_labels = torch.tensor([0 for _ in range(len(nodule_ids))])
                nodule_luna25_labels = torch.tensor([0 for _ in range(len(nodule_ids))])
                nodule_has_cancer_labels = torch.tensor(
                    [0 for _ in range(len(nodule_ids))]
                )
                nodule_has_luna25_labels = torch.tensor(
                    [0 for _ in range(len(nodule_ids))]
                )

            num_nodules = len(nodule_ids)
            segmentation_ = sparse_segmentation.to_dense()

            # resize segmentation_
            segmentation_ = segmentation_.permute(2, 0, 1).unsqueeze(1)
            segmentation_ = F.interpolate(
                segmentation_,
                size=(self.args.img_size[0], self.args.img_size[1]),
                mode="nearest-exact",
            )
            # Remove batch/channel dims and permute back: (H, W, D)
            segmentation_ = segmentation_.squeeze(1).permute(1, 2, 0)
        else:
            segmentation = segmentation_ = None
            nodule_volumes = torch.tensor([0])
            nodule_cancer_labels = torch.tensor([0])
            nodule_luna25_labels = torch.tensor([0])
            nodule_has_cancer_labels = torch.tensor([0])
            nodule_has_luna25_labels = torch.tensor([0])
            nodule_ids = torch.tensor([0])
            num_nodules = 0

        return (
            segmentation,
            segmentation_,
            nodule_volumes,
            nodule_cancer_labels,
            nodule_luna25_labels,
            nodule_has_cancer_labels,
            nodule_has_luna25_labels,
            nodule_ids,
            num_nodules,
        )

    def process_nodules(self, sample, label, nodule_data):
        affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))
        pid_series_key = "{}_{}".format(sample["pid"], sample["series"])
        (
            _,
            segmentation_,
            nodule_volumes,
            nodule_cancer_labels,
            nodule_luna25_labels,
            nodule_has_cancer_labels,
            nodule_has_luna25_labels,
            nodule_ids,
            num_nodules,
        ) = nodule_data

        new_nodule_ids = []
        new_nodule_indices = []
        old_nodule_ids = []
        nodule_confidence = []
        has_confidence = (sample["confidence_path"] is not None) and os.path.exists(
            sample["confidence_path"]
        )
        if has_confidence and (
            (sample["segmentation_path"] is not None)
            or (pid_series_key in self.nlst_luna25)
        ):
            confidence = torch.load(sample["confidence_path"], weights_only=False)
            confidence = torch.softmax(confidence["logit"], -1)[:, -1]

            segmentation_ = MetaTensor(
                segmentation_,
                affine=affine,
                dtype=torch.float32,
            )
            for ncount, nodule_id in enumerate(nodule_ids):
                if confidence[ncount] < self.args.min_nodule_confidence:
                    continue
                nodule_mask = (segmentation_ == nodule_id) * 1.0
                nodule_mask = self.augmentations.segmentation_transform(nodule_mask)
                if nodule_mask.sum() == 0:
                    continue
                new_nodule_ids.append(len(new_nodule_ids) + 1)
                new_nodule_indices.append(ncount)
                old_nodule_ids.append(nodule_id.item())
                label = label + (nodule_mask > 0) * new_nodule_ids[-1]
            nodule_ids = torch.tensor(new_nodule_ids)
            new_nodule_indices = torch.tensor(new_nodule_indices)
            old_nodule_ids = torch.tensor(old_nodule_ids)
            num_nodules = len(nodule_ids)
            if num_nodules == 0:
                nodule_volumes = torch.tensor([])
                nodule_cancer_labels = torch.tensor([])
                nodule_luna25_labels = torch.tensor([])
                nodule_has_cancer_labels = torch.tensor([])
                nodule_has_luna25_labels = torch.tensor([])
                nodule_ids = torch.tensor([])
                nodule_confidence = torch.tensor([])
                old_nodule_ids = torch.tensor([])
                new_nodule_indices = torch.tensor([])
            else:
                nodule_volumes = nodule_volumes[new_nodule_indices]
                nodule_cancer_labels = nodule_cancer_labels[new_nodule_indices]
                nodule_luna25_labels = nodule_luna25_labels[new_nodule_indices]
                nodule_has_cancer_labels = nodule_has_cancer_labels[new_nodule_indices]
                nodule_has_luna25_labels = nodule_has_luna25_labels[new_nodule_indices]
                nodule_confidence = confidence[new_nodule_indices]
        else:
            nodule_volumes = torch.tensor([])
            nodule_cancer_labels = torch.tensor([])
            nodule_luna25_labels = torch.tensor([])
            nodule_has_cancer_labels = torch.tensor([])
            nodule_has_luna25_labels = torch.tensor([])
            nodule_ids = torch.tensor([])
            nodule_confidence = torch.tensor([])
            old_nodule_ids = torch.tensor([])
            new_nodule_indices = torch.tensor([])

        return (
            nodule_ids,
            label,
            new_nodule_indices,
            old_nodule_ids,
            nodule_volumes,
            nodule_cancer_labels,
            nodule_luna25_labels,
            nodule_has_cancer_labels,
            nodule_has_luna25_labels,
            num_nodules,
            nodule_confidence,
        )

    def _pad_along_depth(self, tensor: torch.Tensor, target_depth: int) -> torch.Tensor:
        """
        Pad a (C, D, H, W) or (D, H, W) tensor symmetrically along D to target_depth.
        """
        if tensor.dim() == 3:
            depth = tensor.shape[0]
            if depth < target_depth:
                pad_total = target_depth - depth
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                return F.pad(tensor, (0, 0, 0, 0, pad_left, pad_right))
            elif depth > target_depth:
                crop_total = depth - target_depth
                crop_left = crop_total // 2
                crop_right = crop_total - crop_left
                if crop_right == 0:
                    return tensor[crop_left:]
                return tensor[crop_left:-crop_right]
            return tensor
        elif tensor.dim() == 4:
            depth = tensor.shape[1]
            if depth < target_depth:
                pad_total = target_depth - depth
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                return F.pad(tensor, (0, 0, 0, 0, pad_left, pad_right))
            elif depth > target_depth:
                crop_total = depth - target_depth
                crop_left = crop_total // 2
                crop_right = crop_total - crop_left
                if crop_right == 0:
                    return tensor[:, crop_left:]
                return tensor[:, crop_left:-crop_right]
            return tensor
        else:
            raise ValueError(
                f"Unexpected tensor shape for D padding: {tuple(tensor.shape)}"
            )

    def process_item(self, sample):
        try:
            if sample["split"] == "test":
                examid = "{}{}{}".format(
                    sample["pid"],
                    int(sample["screen_timepoint"]),
                    int(sample["series"].split(".")[-1][-3:]),
                )
            else:
                examid = sample["exam"]
            rve_sample = f"/path/to/local/cache/nlst_rve2/{examid}.1.0"
            if not os.path.exists(rve_sample):
                return

            image = rve.load_sample(rve_sample, use_hardware_acceleration=False)[None]
            # pad
            image = self._pad_along_depth(image, 256)
            # windowing
            image = self.process_image_for_pillar(image)
            item = {
                "x": image,
                "y": sample["y"],
                "has_y": 1,
                "has_mask": int(sample["segmentation_path"] is not None),
                "cancer_laterality": sample["cancer_laterality"],
                "has_laterality_class": 1,
                "exam": sample["exam"],
                "pid": sample["pid"],
                "dataset": "nlst",
                "time_at_event": sample["time_at_event"],
                "y_seq": sample["y_seq"],
                "y_mask": sample["y_mask"],
                "anatomy": "chest_ct",
            }
            return item

        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

    def process_item_(self, sample):
        try:
            image = self.load_image(sample)
            if image is None:
                return
            affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))

            if not self.args.use_pillar_processing:
                # window first
                image = self.process_image_default(image)

            # lung_mask = np.load(
            #     "/path/to/local/cache/lung_ct/nlst_lung_mask/sample_{}.npy".format(
            #         sample["exam"]
            #     )
            # )
            # # ! FIX
            # if lung_mask.max() == 0:
            #     lung_mask = np.ones_like(lung_mask)
            # lung_mask = (lung_mask > 0) * 1.0
            # lung_mask = MetaTensor(
            #     np.transpose(lung_mask, (1, 2, 0)),
            #     affine=affine,
            #     dtype=torch.float32,
            # )
            image_dict = {
                "image": image,
                # "lung": lung_mask,
            }
            image_dict = self.augmentations(image_dict)

            if self.args.use_pillar_processing:
                # augmentations first: spacing, pad2d, clip, pad3d
                # apply all windows + minmax
                image = self.process_image_for_pillar(image_dict["image"])
                image_dict["image"] = image

            x = image_dict["image"].permute(0, 3, 1, 2)
            item = {
                "x": x,
                "y": sample["y"],
                "has_y": 1,
                "has_mask": int(sample["segmentation_path"] is not None),
                "cancer_laterality": sample["cancer_laterality"],
                "has_laterality_class": 1,
                "exam": sample["exam"],
                "pid": sample["pid"],
                "dataset": "nlst",
                "time_at_event": sample["time_at_event"],
                "y_seq": sample["y_seq"],
                "y_mask": sample["y_mask"],
                "anatomy": "chest_ct",
            }

            label = torch.zeros(
                image_dict["image"].shape[-3:],
                dtype=image_dict["image"].dtype,
                device=image_dict["image"].device,
            ).unsqueeze(0)

            # load the segmentation
            segmentation_metadata = self.load_segmentation(sample)

            nodule_data = self.process_nodules(sample, label, segmentation_metadata)
            (
                nodule_ids,
                label,
                new_nodule_indices,
                old_nodule_ids,
                nodule_volumes,
                nodule_cancer_labels,
                nodule_luna25_labels,
                nodule_has_cancer_labels,
                nodule_has_luna25_labels,
                num_nodules,
                nodule_confidence,
            ) = nodule_data

            label = label.permute(0, 3, 1, 2)
            x = image_dict["image"].permute(0, 3, 1, 2)
            try:
                mask = label.as_tensor().contiguous().to(torch.int64)
            except:
                mask = label.contiguous().to(torch.int64)
            if getattr(self.args, "add_segmentation_to_input", False):
                x = torch.cat([x, label], dim=0)

            item = {
                "x": x,
                "mask": mask,
                "lung": image_dict["lung"].permute(0, 3, 1, 2),
                "y": sample["y"],
                "has_y": 1,
                "has_mask": int(sample["segmentation_path"] is not None),
                "cancer_laterality": sample["cancer_laterality"],
                "has_laterality_class": 1,
                "exam": sample["exam"],
                "pid": sample["pid"],
                "dataset": "nlst",
                "num_nodules": num_nodules,
                "nodule_ids": nodule_ids,
                "nodule_has_cancer_labels": nodule_has_cancer_labels,
                "nodule_has_luna25_labels": nodule_has_luna25_labels,
                "nodule_volumes": nodule_volumes,
                "nodule_cancer_labels": nodule_cancer_labels,
                "nodule_luna25_labels": nodule_luna25_labels,
                "time_at_event": sample["time_at_event"],
                "y_seq": sample["y_seq"],
                "y_mask": sample["y_mask"],
                "anatomy": "chest_ct",
            }

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

class NLST_Nodule_Segmentation(NLST_Survival_Dataset2):
    def __init__(self, args, split_group):
        """
        NLST Dataset
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(NLST_Survival_Dataset, self).__init__()

        self.split_group = split_group
        self.args = args
        args.augment_before_cropping = True
        self.augmentations = FullAugmentations(args, split=split_group)
        self.patch_augmentations = PatchAugmentations(args, split=split_group)
        self.pad = args.anatomix_pad_size
        self.anatomix_crop_size = args.anatomix_crop_size

        try:
            self.metadata_json = json.load(open(args.dataset_file_path, "r"))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

        if self.args.region_annotations_filepath:
            self.annotations_metadata = json.load(
                open(self.args.region_annotations_filepath, "r")
            )
            self.annotations_metadata["exams"] = set(self.annotations_metadata["exams"])
            self.annotations_metadata["lung_masks"] = set(
                self.annotations_metadata["lung_masks"]
            )
        else:
            self.annotations_metadata = {}

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        print(self.get_summary_statement(self.dataset, split_group))

        self.metadata_json = None

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

        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row.get("split", "train"),
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            if (split != split_group) and not self.args.turn_off_splits:
                continue

            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():
                    series_dict["series_data"] = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in series_dict["series_data"].items()
                    }
                    if self.skip_sample(series_id, series_dict, exam_dict, pt_metadata):
                        continue

                    sample = self.get_volume_dict(
                        series_id,
                        series_dict,
                        exam_dict,
                        pt_metadata,
                        pid,
                        split,
                    )
                    if len(sample) == 0:
                        continue

                    if isinstance(sample, list):
                        dataset.extend(sample)
                    else:
                        dataset.append(sample)

        return dataset

    def skip_sample(self, series_id, series_dict, exam_dict, pt_metadata):
        # skip if no lung mask
        examid = get_examid(
            pt_metadata["pid"][0], exam_dict["screen_timepoint"], series_id
        )

        if examid not in self.annotations_metadata["lung_masks"]:
            return True

        if examid not in self.annotations_metadata["exams"]:
            return True

        if examid not in self.annotations_metadata["confidence_files"]:
            return True

        return False

    def get_volume_dict(
        self,
        series_id,
        series_dict,
        exam_dict,
        pt_metadata,
        pid,
        split,
    ):
        img_paths = series_dict["paths"]
        slice_locations = series_dict["img_position"]
        series_data = series_dict["series_data"]
        device = DEVICE_ID[DEVICE_TO_NAME[series_data["manufacturer"][0]]]
        screen_timepoint = series_data["study_yr"][0]
        assert screen_timepoint == exam_dict["screen_timepoint"]

        if img_paths[-1].endswith(".png"):
            img_paths = [
                (
                    PID2DICOM_DIRECTORY[pid]
                    + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                ).replace(".png", ".dcm")
                for path in img_paths
            ]

        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations
        )

        examid = "{}{}{}{}".format(
            pid,
            int(screen_timepoint),
            series_id.split(".")[-1][:5],
            series_id.split(".")[-1][-5:],
        )
        segpath = None
        if examid in self.annotations_metadata["exams"]:
            segpath = f"/path/to/local/cache/lung_ct/nlst_abnormalities51_nnunet_sparse_segmentation/sample_{examid}.pt"

        sample = {
            "paths": sorted_img_paths,
            "slice_locations": sorted_slice_locs,
            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
            "exam": examid,
            "accession": exam_dict["accession_number"],
            "series": series_id,
            "screen_timepoint": screen_timepoint,
            "pid": pid,
            "num_original_slices": len(series_dict["paths"]),
            "pixel_spacing": series_dict["pixel_spacing"]
            + [series_dict["slice_thickness"]],
            "segmentation_path": segpath,
            "has_annotation": int(segpath is not None),
            "lung_mask_path": f"/path/to/local/cache/lung_ct/nlst_lung_mask/sample_{examid}.npy",
            "confidence_path": self.annotations_metadata["confidence_files"][examid],
        }

        return sample

    def process_item(self, sample):
        if self.args.use_full_volume:
            return self.process_full_item(sample)
        else:
            return self.process_patch_item(sample)

    def process_full_item(self, sample):
        try:
            # load the nifti if it exists, otherwise convert from dicom
            path = self.annotations_metadata["nifti_files"].get(sample["exam"], None)
            if path is not None:
                try:
                    nifti_img = sitk.ReadImage(path)
                    image = sitk.GetArrayFromImage(nifti_img)  # z, y, x
                    image = image.transpose(1, 2, 0)  # y, x, z
                except:
                    try:
                        path = f"/path/to/local/cache/lung_ct/nlst_nifti/sample_{sample['exam']}.nii.gz"
                        image = pydicom_to_nifti(
                            sample["paths"], path, save_nifti=True
                        )  # y, x, z
                        self.annotations_metadata["nifti_files"][sample["exam"]] = path
                    except:
                        return
            else:
                try:
                    path = f"/path/to/local/cache/lung_ct/nlst_nifti/sample_{sample['exam']}.nii.gz"
                    image = pydicom_to_nifti(
                        sample["paths"], path, save_nifti=True
                    )  # y, x, z
                    self.annotations_metadata["nifti_files"][sample["exam"]] = path
                except:
                    return

            segmentation = torch.load(sample["segmentation_path"])
            sparse_segmentation = segmentation["sparse_segmentation"]
            nodule_volumes = segmentation["nodule_volumes"]
            nodule_ids = segmentation["nodule_ids"]
            num_nodules = len(nodule_ids)
            segmentation_ = sparse_segmentation.to_dense()

            # resize segmentation_
            segmentation_ = segmentation_.permute(2, 0, 1).unsqueeze(1)
            segmentation_ = F.interpolate(
                segmentation_,
                size=(self.args.img_size[0], self.args.img_size[1]),
                mode="nearest-exact",
            )
            # Remove batch/channel dims and permute back: (H, W, D)
            segmentation_ = segmentation_.squeeze(1).permute(1, 2, 0)

            affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))
            image = apply_windowing(image.astype(np.float64), -600, 1600)
            image = image // 256
            image = MetaTensor(
                image,
                affine=affine,
                dtype=torch.float32,
            )
            lung_mask = np.load(
                "/path/to/local/cache/lung_ct/nlst_lung_mask/sample_{}.npy".format(
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
                "lung": lung_mask,
            }
            image_dict = self.augmentations(image_dict)

            label = torch.zeros_like(image_dict["image"])

            segmentation_ = MetaTensor(
                segmentation_,
                affine=affine,
                dtype=torch.float32,
            )

            confidence = torch.load(sample["confidence_path"], weights_only=False)
            confidence = torch.softmax(confidence["logit"], -1)[:, -1]

            new_nodule_ids = []
            for ncount, nodule_id in enumerate(nodule_ids):
                if confidence[ncount] < self.args.min_nodule_confidence:
                    continue
                nodule_mask = (segmentation_ == nodule_id) * 1.0
                nodule_mask = self.augmentations.segmentation_transform(nodule_mask)
                if nodule_mask.sum() == 0:
                    continue
                new_nodule_ids.append(len(new_nodule_ids) + 1)
                label = label + (nodule_mask > 0) * new_nodule_ids[-1]
            nodule_ids = torch.tensor(new_nodule_ids)
            num_nodules = len(nodule_ids)
            if num_nodules == 0:
                nodule_volumes = torch.tensor([0])
                nodule_ids = torch.tensor([0])

            label = label.permute(0, 3, 1, 2)

            item = {
                "x": image_dict["image"].permute(0, 3, 1, 2),
                "mask": label.as_tensor().contiguous().to(torch.int64),
                "lung": image_dict["lung"].permute(0, 3, 1, 2),
                "has_mask": label.sum() > 0,
                "exam": sample["exam"],
                "pid": sample["pid"],
                "dataset": "nlst",
                "num_nodules": num_nodules,
                "nodule_ids": nodule_ids,
                "nodule_volumes": nodule_volumes,
            }

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

    def process_patch_item(self, sample):
        # load the nifti if it exists, otherwise convert from dicom
        path = self.annotations_metadata["nifti_files"].get(sample["exam"], None)
        if path is not None:
            try:
                nifti_img = sitk.ReadImage(path)
                image = sitk.GetArrayFromImage(nifti_img)  # z, y, x
                image = image.transpose(1, 2, 0)  # y, x, z
            except:
                try:
                    path = f"/path/to/local/cache/lung_ct/nlst_nifti/sample_{sample['exam']}.nii.gz"
                    image = pydicom_to_nifti(
                        sample["paths"], path, save_nifti=True
                    )  # y, x, z
                    self.annotations_metadata["nifti_files"][sample["exam"]] = path
                except:
                    return
        else:
            try:
                path = f"/path/to/local/cache/lung_ct/nlst_nifti/sample_{sample['exam']}.nii.gz"
                image = pydicom_to_nifti(
                    sample["paths"], path, save_nifti=True
                )  # y, x, z
                self.annotations_metadata["nifti_files"][sample["exam"]] = path
            except:
                return

        image = apply_windowing(image.astype(np.float64), -600, 1600)
        image = image // 256
        image = MetaTensor(
            image,
            affine=torch.diag(torch.tensor(sample["pixel_spacing"] + [1])),
            dtype=torch.float32,
        )
        # load the lung mask
        # lung_mask = np.load(
        #     "/path/to/local/cache/lung_ct/nlst_lung_mask/sample_{}.npy".format(
        #         sample["exam"]
        #     )
        # )
        # lung_mask = (lung_mask > 0) * 1.0
        # lung_mask = MetaTensor(
        #     np.transpose(lung_mask, (1, 2, 0)),
        #     affine=torch.diag(torch.tensor(sample["pixel_spacing"] + [1])),
        #     dtype=torch.float32,
        # )

        # load the  segmentation
        segmentation = torch.load(sample["segmentation_path"])
        sparse_segmentation = segmentation["sparse_segmentation"]
        sparse_segmentation = sparse_segmentation.coalesce()
        nodule_volumes = segmentation["nodule_volumes"]
        nodule_ids = segmentation["nodule_ids"]
        num_nodules = len(nodule_ids)
        segmentation_ = sparse_segmentation.to_dense()

        confidence = torch.load(sample["confidence_path"], weights_only=False)
        confidence = torch.softmax(confidence["logit"], -1)[:, -1]
        nodule_ids = nodule_ids[confidence >= self.args.min_nodule_confidence]

        # choose a random nodule
        if len(nodule_ids) == 0:
            return
        chosen_nodule_idx = np.random.randint(0, len(nodule_ids))
        chosen_nodule_id = nodule_ids[chosen_nodule_idx]

        # get coordinates
        ys, xs, zs = sparse_segmentation.indices()[
            :, sparse_segmentation.values() == chosen_nodule_id
        ]
        box = {
            "y_start": ys.min().item(),
            "y_stop": ys.max().item() + 1,
            "x_start": xs.min().item(),
            "x_stop": xs.max().item() + 1,
            "z_start": zs.min().item(),
            "z_stop": zs.max().item() + 1,
        }

        # Ensure image and label have the correct spatial size (args.img_size)
        img_size = self.args.img_size  # (H, W)
        img_h, img_w = image.shape[0], image.shape[1]
        if (img_h, img_w) != tuple(img_size):
            # image: (H, W, D), label: (H, W, D)
            # Add batch and channel dims for interpolation: (1, 1, D, H, W)
            image_ = image.permute(2, 0, 1).unsqueeze(1)
            segmentation_ = segmentation_.permute(2, 0, 1).unsqueeze(1)
            # lung_mask = lung_mask.permute(2, 0, 1).unsqueeze(1)
            image_ = F.interpolate(
                image_,
                size=(img_size[0], img_size[1]),
                mode="bilinear",
                align_corners=False,
            )
            segmentation_ = F.interpolate(
                segmentation_, size=(img_size[0], img_size[1]), mode="nearest-exact"
            )
            # lung_mask = F.interpolate(
            #     lung_mask, size=(img_size[0], img_size[1]), mode="nearest-exact"
            # )
            # Remove batch/channel dims and permute back: (H, W, D)
            image = image_.squeeze(1).permute(1, 2, 0)
            label = segmentation_.squeeze(1).permute(1, 2, 0)
            # lung_mask = lung_mask.squeeze(1).permute(1, 2, 0)

            # adjust the box to the new image size
            box["y_start"] = int(box["y_start"] * img_size[0] / 1024)
            box["y_stop"] = int(box["y_stop"] * img_size[0] / 1024)
            box["x_start"] = int(box["x_start"] * img_size[1] / 1024)
            box["x_stop"] = int(box["x_stop"] * img_size[1] / 1024)

        if (self.split_group != "train") and self.args.predict:
            sample["image"] = image[None]
            sample["label"] = label[None]
            y = 1
            item = {
                "x": sample["image"].permute(0, 3, 1, 2),
                "mask": sample["label"].permute(0, 3, 1, 2),
                "y": y,
                "exam": sample["exam"],
                "pid": sample["pid"],
                "dataset": "nlst",
            }
        else:
            h1, w1, _ = self.anatomix_crop_size
            d1 = self.pad[-1]
            # augment first before cropping
            image_sample = self.patch_augmentations(
                {
                    "image": image,
                    "label": label,
                }
            )
            cbbox = random_pad_3d_box(
                box,
                image,
                min_height=h1,
                min_width=w1,
                min_depth=d1,
                random_hw=True,
                random_d=True,
            )
            image_sample["image"] = image_sample["image"][0][cbbox][None]
            image_sample["label"] = image_sample["label"][0][cbbox][None]

            image_sample = self.patch_augmentations._transform(image_sample)

            item = {
                "x": image_sample["image"].permute(0, 3, 1, 2),
                "mask": image_sample["label"].permute(0, 3, 1, 2),
                "exam": sample["exam"],
                "pid": sample["pid"],
                "dataset": "nlst",
            }
        return item

    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed NLST CT Cancer Risk {} dataset with {} records, {} exams, {} patients"
        exams = set([d["exam"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group,
            len(dataset),
            len(exams),
            len(patients),
        )
        return statement

class NLST_Nodule_PreprocessedPatches_Segmentation(NLST_Nodule_Segmentation):
    def __init__(self, args, split_group):
        """
        NLST Dataset
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(NLST_Survival_Dataset, self).__init__()

        self.split_group = split_group
        self.args = args

        try:
            self.metadata_json = json.load(open(args.dataset_file_path, "r"))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

        self.exam_to_cached_nodules = pickle.load(
            open(
                "/data/rbg/shared/datasets/NLST/NLST/nlst_exam_to_nodules_abn51.p", "rb"
            )
        )
        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        print(self.get_summary_statement(self.dataset, split_group))

        self.metadata_json = None

    def skip_sample(self, series_id, series_dict, exam_dict, pt_metadata):
        examid = get_examid(
            pt_metadata["pid"][0], exam_dict["screen_timepoint"], series_id
        )
        if examid not in self.exam_to_cached_nodules:
            return True
        return False

    def get_volume_dict(
        self,
        series_id,
        series_dict,
        exam_dict,
        pt_metadata,
        pid,
        split,
    ):
        img_paths = series_dict["paths"]
        slice_locations = series_dict["img_position"]
        series_data = series_dict["series_data"]
        device = DEVICE_ID[DEVICE_TO_NAME[series_data["manufacturer"][0]]]
        screen_timepoint = series_data["study_yr"][0]
        assert screen_timepoint == exam_dict["screen_timepoint"]

        if img_paths[-1].endswith(".png"):
            img_paths = [
                (
                    PID2DICOM_DIRECTORY[pid]
                    + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                ).replace(".png", ".dcm")
                for path in img_paths
            ]

        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations
        )

        examid = "{}{}{}{}".format(
            pid,
            int(screen_timepoint),
            series_id.split(".")[-1][:5],
            series_id.split(".")[-1][-5:],
        )

        samples = []
        for nid in self.exam_to_cached_nodules[examid]:
            samples.append(
                {
                    "paths": sorted_img_paths,
                    "slice_locations": sorted_slice_locs,
                    "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
                    "exam": examid,
                    "accession": exam_dict["accession_number"],
                    "series": series_id,
                    "screen_timepoint": screen_timepoint,
                    "pid": pid,
                    "num_original_slices": len(series_dict["paths"]),
                    "pixel_spacing": series_dict["pixel_spacing"]
                    + [series_dict["slice_thickness"]],
                    "segmentation_path": f"/path/to/local/cache/lung_ct/nlst_abnormalities51_patches/{examid}_nodule{nid}.pt",
                    "lung_mask_path": f"/path/to/local/cache/lung_ct/nlst_lung_mask/sample_{examid}.npy",
                }
            )

        return samples

    def process_item(self, sample):
        try:
            x, segmentation = torch.load(
                sample["segmentation_path"], weights_only=False
            )
            item = {
                "x": x[None],
                "mask": (segmentation[None] > 0) * 1.0,
                "exam": sample["exam"],
                "pid": sample["pid"],
                "dataset": "nlst",
            }
            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

class NLST_LUNA25(NLST_Survival_Dataset2):
    def create_dataset(self, split_group):
        dataset = []
        # add exams with abnormalities
        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row.get("split", "train"),
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )
            if (split != split_group) and not self.args.turn_off_splits:
                continue
            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():
                    series_dict["series_data"] = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in series_dict["series_data"].items()
                    }
                    if f"{pid}_{series_id}" not in self.nlst_luna25:
                        continue
                    sample = self.get_volume_dict(
                        series_id,
                        series_dict,
                        exam_dict,
                        pt_metadata,
                        pid,
                        split,
                    )
                    if len(sample) == 0:
                        continue

                    if isinstance(sample, list):
                        dataset.extend(sample)
                    else:
                        dataset.append(sample)
        return dataset

    def get_volume_dict(
        self,
        series_id,
        series_dict,
        exam_dict,
        pt_metadata,
        pid,
        split,
    ):
        series_data = series_dict["series_data"]
        screen_timepoint = series_data["study_yr"][0]
        assert screen_timepoint == exam_dict["screen_timepoint"]

        examid = "{}{}{}{}".format(
            pid,
            int(screen_timepoint),
            series_id.split(".")[-1][:5],
            series_id.split(".")[-1][-5:],
        )
        samples = []
        for nid, nodule_dict in enumerate(self.nlst_luna25[f"{pid}_{series_id}"]):
            y = nodule_dict["nodule_label"]
            samples.append(
                {
                    "split": split,
                    "path": "/path/to/local/cache/lung_ct/nlst_luna25_patches/{}_{}_{}.npy".format(
                        pid, series_id, nid
                    ),
                    "y": int(y),
                    "exam": f"{examid}-{nid}",
                    "pid": pid,
                    "time_at_event": 0,
                    "series": series_id,
                    "screen_timepoint": screen_timepoint,
                }
            )
        return samples

    def process_item(self, sample):
        try:
            if self.args.use_pillar_processing:
                if sample["split"] == "test":
                    examid = "{}{}{}".format(
                        sample["pid"],
                        int(sample["screen_timepoint"]),
                        int(sample["series"].split(".")[-1][-3:]),
                    )
                else:
                    examid = sample["exam"].split("-")[0]
                rve_sample = f"/path/to/local/cache/nlst_rve2/{examid}.1.0"
                if not os.path.exists(rve_sample):
                    return

                image = rve.load_sample(rve_sample, use_hardware_acceleration=False)[
                    None
                ]
                # pad
                image = self._pad_along_depth(image, 256)
                # windowing
                x = self.process_image_for_pillar(image)

            else:
                image = np.load(sample["path"])
                if image is None:
                    return
                image = torch.from_numpy(image).float()

                # window first
                image = self.process_image_default(image).unsqueeze(0)
                image = image.permute(0, 3, 1, 2)  # B, D, H, W
                image = F.interpolate(
                    image,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # B, H, W, D

                image_dict = {
                    "image": image,
                }
                image_dict = self.patch_augmentations._transform(image_dict)
                image_dict = self.patch_augmentations.predict_transforms.transforms[-1](
                    image_dict
                )

                x = image_dict["image"].permute(0, 3, 1, 2)
            item = {
                "x": x,
                "y": sample["y"],
                "exam": sample["exam"],
                "anatomy": "chest_ct",
            }
            return item

        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

class NLST_LUNA25_Annotated_Cancers(NLST_Survival_Dataset2):
    def create_dataset(self, split_group):
        nlst_luna25_series = list(k.split("_")[-1] for k in self.nlst_luna25)
        annotated_cancers = list(self.annotations_metadata.keys())
        series_to_use = set(nlst_luna25_series + annotated_cancers)

        dataset = []

        # add exams with abnormalities
        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row.get("split", "train"),
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )
            if (split != split_group) and not self.args.turn_off_splits:
                continue

            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():
                    series_dict["series_data"] = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in series_dict["series_data"].items()
                    }
                    if series_id not in series_to_use:
                        continue

                    sample = self.get_volume_dict(
                        series_id,
                        series_dict,
                        exam_dict,
                        pt_metadata,
                        pid,
                        split,
                    )
                    if len(sample) == 0:
                        continue
                    if isinstance(sample, list):
                        dataset.extend(sample)
                    else:
                        dataset.append(sample)

        return dataset

class NLST_Sparse_Confidence(NLST_Survival_Dataset2):
    def __init__(self, args, split_group):
        super().__init__(args, split_group)
        self.augmentations = PatchAugmentations(args, split=split_group)
        self.pad = args.anatomix_pad_size
        self.anatomix_crop_size = args.anatomix_crop_size

    def skip_sample(self, series_id, series_dict, exam_dict, pt_metadata):
        # skip if no lung mask
        examid = get_examid(
            pt_metadata["pid"][0], exam_dict["screen_timepoint"], series_id
        )

        if examid not in self.annotations_metadata["lung_masks"]:
            return True

        if examid not in self.annotations_metadata["exams"]:
            return True

        return False

    def get_volume_dict(
        self,
        series_id,
        series_dict,
        exam_dict,
        pt_metadata,
        pid,
        split,
    ):
        img_paths = series_dict["paths"]
        slice_locations = series_dict["img_position"]
        series_data = series_dict["series_data"]
        screen_timepoint = series_data["study_yr"][0]
        assert screen_timepoint == exam_dict["screen_timepoint"]

        if img_paths[-1].endswith(".png"):
            img_paths = [
                (
                    PID2DICOM_DIRECTORY[pid]
                    + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                ).replace(".png", ".dcm")
                for path in img_paths
            ]

        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations
        )

        examid = "{}{}{}{}".format(
            pid,
            int(screen_timepoint),
            series_id.split(".")[-1][:5],
            series_id.split(".")[-1][-5:],
        )
        segpath = None
        if examid in self.annotations_metadata["exams"]:
            segpath = f"/path/to/local/cache/lung_ct/nlst_abnormalities51_nnunet_sparse_segmentation/sample_{examid}.pt"

        sample = {
            "paths": sorted_img_paths,
            "slice_locations": sorted_slice_locs,
            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
            "exam": examid,
            "accession": exam_dict["accession_number"],
            "series": series_id,
            "screen_timepoint": screen_timepoint,
            "pid": pid,
            "num_original_slices": len(series_dict["paths"]),
            "pixel_spacing": series_dict["pixel_spacing"]
            + [series_dict["slice_thickness"]],
            "segmentation_path": segpath,
            "has_annotation": int(segpath is not None),
            "lung_mask_path": f"/path/to/local/cache/lung_ct/nlst_lung_mask/sample_{examid}.npy",
        }

        if self.args.use_risk_factors:
            sample["risk_factors"] = self.get_risk_factors(
                pt_metadata, screen_timepoint, return_dict=False
            )

        return sample

    def process_item(self, sample):
        try:
            h1, w1, _ = self.anatomix_crop_size
            d1 = self.pad[-1]
            # load the nifti if it exists, otherwise convert from dicom
            # path = self.annotations_metadata["nifti_files"].get(sample["exam"], None)
            path = (
                f"/path/to/local/cache/lung_ct/nlst_nifti/sample_{sample['exam']}.nii.gz"
            )
            if path is not None:
                try:
                    nifti_img = sitk.ReadImage(path)
                    image = sitk.GetArrayFromImage(nifti_img)  # z, y, x
                    image = image.transpose(1, 2, 0)  # y, x, z
                except:
                    try:
                        path = f"/path/to/local/cache/lung_ct/nlst_nifti/sample_{sample['exam']}.nii.gz"
                        image = pydicom_to_nifti(
                            sample["paths"], path, save_nifti=True
                        )  # y, x, z
                        self.annotations_metadata["nifti_files"][sample["exam"]] = path
                    except:
                        return
            else:
                try:
                    path = f"/path/to/local/cache/lung_ct/nlst_nifti/sample_{sample['exam']}.nii.gz"
                    image = pydicom_to_nifti(
                        sample["paths"], path, save_nifti=True
                    )  # y, x, z
                    self.annotations_metadata["nifti_files"][sample["exam"]] = path
                except:
                    return

            affine = torch.diag(torch.tensor(sample["pixel_spacing"] + [1]))
            image = apply_windowing(image.astype(np.float64), -600, 1600)
            image = image // 256
            image = MetaTensor(
                image,
                affine=affine,
                dtype=torch.float32,
            )

            # Ensure image and label have the correct spatial size (args.img_size)
            img_size = self.args.img_size  # (H, W)
            img_h, img_w = image.shape[0], image.shape[1]
            if (img_h, img_w) != tuple(img_size):
                # image: (H, W, D), label: (H, W, D)
                # Add batch and channel dims for interpolation: (1, 1, D, H, W)
                image_ = image.permute(2, 0, 1).unsqueeze(1)
                image_ = F.interpolate(
                    image_,
                    size=(img_size[0], img_size[1]),
                    mode="bilinear",
                    align_corners=False,
                )

                # Remove batch/channel dims and permute back: (H, W, D)
                image = image_.squeeze(1).permute(1, 2, 0)

            # load the segmentation
            segmentation = torch.load(sample["segmentation_path"])
            sparse_segmentation = segmentation["sparse_segmentation"]
            nodule_volumes = segmentation["nodule_volumes"]
            nodule_ids = segmentation["nodule_ids"]
            num_nodules = len(nodule_ids)
            segmentation = sparse_segmentation.to_dense()

            inputs = []
            for nodid in nodule_ids:
                nodule_mask = (segmentation == nodid) * 1.0
                if nodule_mask.sum() == 0:
                    continue
                label = MetaTensor(
                    nodule_mask,
                    affine=affine,
                    dtype=torch.float32,
                )
                ys, xs, zs = sparse_segmentation.coalesce().indices()[
                    :, sparse_segmentation.coalesce().values() == nodid
                ]
                box = {
                    "z_start": int(zs.min()),
                    "z_stop": int(zs.max()) + 1,
                    "y_start": int(ys.min()),
                    "y_stop": int(ys.max()) + 1,
                    "x_start": int(xs.min()),
                    "x_stop": int(xs.max()) + 1,
                }
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
                sample = self.augmentations.test_transforms(sample)

                sample["image"] = sample["image"].permute(0, 3, 1, 2)
                sample["label"] = sample["label"].permute(0, 3, 1, 2)
                x = torch.cat([sample["image"], sample["label"]], dim=0)
                inputs.append(x)

            item = {
                "x": torch.stack(inputs),
                "exam": sample["exam"],
                "pid": sample["pid"],
                "nodule_ids": nodule_ids,
                "num_nodules": num_nodules,
                "dataset": "nlst",
            }

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed NLST CT Cancer Risk {} dataset with {} records, {} exams, {} patients\n"
        exams = set([d["exam"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group,
            len(dataset),
            len(exams),
            len(patients),
        )
        return statement

class NLST_Sparse_Confidence_SybilTest(NLST_Sparse_Confidence):
    def __init__(self, args, split_group):
        self.sybil_test_exams = set(
            os.listdir("/path/to/local/cache/lung_ct/sybil15_nlst_test_hiddens/")
        )
        super().__init__(args, split_group)

    def skip_sample(self, series_id, series_dict, exam_dict, pt_metadata):
        examid = get_examid(
            pt_metadata["pid"][0], exam_dict["screen_timepoint"], series_id
        )
        if f"sample_{examid}.predictions" not in self.sybil_test_exams:
            return True

        return False

    def get_volume_dict(
        self,
        series_id,
        series_dict,
        exam_dict,
        pt_metadata,
        pid,
        split,
    ):
        img_paths = series_dict["paths"]
        slice_locations = series_dict["img_position"]
        series_data = series_dict["series_data"]
        screen_timepoint = series_data["study_yr"][0]
        assert screen_timepoint == exam_dict["screen_timepoint"]

        if img_paths[-1].endswith(".png"):
            img_paths = [
                (
                    PID2DICOM_DIRECTORY[pid]
                    + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                ).replace(".png", ".dcm")
                for path in img_paths
            ]

        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations
        )

        examid = "{}{}{}{}".format(
            pid,
            int(screen_timepoint),
            series_id.split(".")[-1][:5],
            series_id.split(".")[-1][-5:],
        )

        segpath = f"/path/to/local/cache/lung_ct/nlst_abnormalities51_nnunet_sparse_segmentation/sample_{examid}.pt"

        sample = {
            "paths": sorted_img_paths,
            "slice_locations": sorted_slice_locs,
            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
            "exam": examid,
            "accession": exam_dict["accession_number"],
            "series": series_id,
            "screen_timepoint": screen_timepoint,
            "pid": pid,
            "num_original_slices": len(series_dict["paths"]),
            "pixel_spacing": series_dict["pixel_spacing"]
            + [series_dict["slice_thickness"]],
            "segmentation_path": segpath,
            "has_annotation": int(segpath is not None),
            "lung_mask_path": f"/path/to/local/cache/lung_ct/nlst_lung_mask/sample_{examid}.npy",
        }

        return sample

class NLST_Longitudinal(NLST_Survival_Dataset2):
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

        test_hiddens = set(
            os.listdir("/path/to/local/cache/lung_ct/sybil15_nlst_test_hiddens/")
        )
        if split_group == "test":
            pid_tracked_nodules_path = "/data/rbg/users/pgmikhael/current/Sybil/notebooks/NLSTNodules/pid2tracked_nodules_test.p"
        else:
            pid_tracked_nodules_path = "/data/rbg/users/pgmikhael/current/Sybil/notebooks/NLSTNodules/pid2tracked_nodules.p"

        pid_tracked_nodules = pickle.load(
            open(
                pid_tracked_nodules_path,
                "rb",
            )
        )
        
        # pid_tracked_nodules is an iterable of (pid, nid2nodes)
        # Build pid2tracked_exams and pid2tracked_nodules ensuring nested dicts
        self.pid2tracked_nodules = {}
        pid2tracked_exams = {}
        for pid, nid2nodes in pid_tracked_nodules:
            # initialize the set of tracked exams for this pid
            pid2tracked_exams.setdefault(pid, defaultdict(set))
            for nid, tps in nid2nodes.items():
                for tp, node in tps.items():
                    pid2tracked_exams[pid][tp].add(node["exam"])
                    # ensure nested dicts exist so we can index to depth 3 safely
                    if pid not in self.pid2tracked_nodules:
                        self.pid2tracked_nodules[pid] = {}
                    if tp not in self.pid2tracked_nodules[pid]:
                        self.pid2tracked_nodules[pid][tp] = {}
                    self.pid2tracked_nodules[pid][tp][node["nodid_in_segmentation"]] = (
                        nid
                    )

        dataset = []
        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row.get("split", "train"),
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            if (split != split_group) and not self.args.turn_off_splits:
                continue

            exams_by_timepoint = {}

            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():
                    series_dict["series_data"] = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in series_dict["series_data"].items()
                    }
                    if split_group == "test":
                        screen_timepoint = exam_dict["screen_timepoint"]
                        examid = "{}{}{}{}".format(
                            pid,
                            int(screen_timepoint),
                            series_id.split(".")[-1][:5],
                            series_id.split(".")[-1][-5:],
                        )
                        if f"sample_{examid}.predictions" not in test_hiddens:
                            continue
                    else:
                        if self.skip_sample(
                            series_id, series_dict, exam_dict, pt_metadata
                        ):
                            continue

                    sample = self.get_volume_dict(
                        series_id,
                        series_dict,
                        exam_dict,
                        pt_metadata,
                        pid,
                        split,
                    )
                    if len(sample) == 0:
                        continue

                    sample["in_luna"] = f"{pid}_{series_id}" in self.nlst_luna25

                    timepoint = sample["screen_timepoint"]
                    examid = sample["exam"]

                    if split_group == "test":
                        exams_by_timepoint[timepoint] = sample
                    else:
                        pid_in_tracked = pid in pid2tracked_exams
                        tp_in_tracked = (
                            timepoint in pid2tracked_exams.get(pid, {})
                            if pid_in_tracked
                            else False
                        )
                        exam_in_tracked = examid in pid2tracked_exams.get(pid, {}).get(
                            timepoint, set()
                        )
                        if (
                            (not pid_in_tracked)
                            or (not tp_in_tracked)
                            or exam_in_tracked
                        ):
                            # if examid in pid2tracked_exams.get(pid, set()):
                            exams_by_timepoint[timepoint] = sample

            for tp1, tp2 in [(0, 1), (1, 2), (0, 2)]:
                if (tp1 in exams_by_timepoint) and (tp2 in exams_by_timepoint):
                    sample = {
                        "pid": pid,
                        "exam1": exams_by_timepoint[tp1],
                        "exam2": exams_by_timepoint[tp2],
                        "y": exams_by_timepoint[tp2]["y"],
                        "time_at_event": exams_by_timepoint[tp2]["time_at_event"],
                    }

                    dataset.append(sample)

            if len(exams_by_timepoint) > 0:
                first_tp = min(exams_by_timepoint)
                sample = {
                    "pid": pid,
                    "exam1": None,
                    "exam2": exams_by_timepoint[first_tp],
                    "y": exams_by_timepoint[first_tp]["y"],
                    "time_at_event": exams_by_timepoint[first_tp]["time_at_event"],
                }

                dataset.append(sample)

        return dataset

    def process_item(self, sample):
        if self.args.precomputed_pillar_hiddens:
            process_exam_fn = self.process_single_cached_exam
        else:
            process_exam_fn = self.process_single_exam
        # NOTE: first item is always the later exam
        pid = sample["pid"]
        item = {
            "anatomy": "chest_ct",
            "dataset": "nlst",
            "pid": pid,
            "exam": sample["exam2"]["exam"],
            "has_prior": int(sample["exam1"] is not None),
        }
        item2 = process_exam_fn(sample["exam2"])
        if item2 is None:
            return None
        num_nodules2 = item2["num_nodules"]
        for key in [
            "x",
            "y",
            "time_at_event",
            "y_seq",
            "y_mask",
            "exam",
            "nodule_volumes",
            "nodule_cancer_labels",
            "nodule_luna25_labels",
            "nodule_has_cancer_labels",
            "nodule_has_luna25_labels",
            "nodule_confidence",
            "old_nodule_ids",
            "nodule_x",
            "logit",
        ]:
            if key in item2:
                item[key] = item2.pop(key)

        item["num_nodules"] = [max(0, item2["num_nodules"])]
        item["nodule_batch_id"] = [0] * self.args.max_num_nodules_per_scan
        item["nodule_tp_id"] = [
            sample["exam2"]["screen_timepoint"]
        ] * self.args.max_num_nodules_per_scan

        if not self.args.precomputed_pillar_hiddens:
            for key in ["mask", "has_mask", "lung"]:
                item[key] = item2.pop(key)

        if sample["exam1"] is not None:
            # item1 was already processed above to collect tracked IDs
            item1 = process_exam_fn(sample["exam1"])
            if item1 is None:
                print(
                    "ERROR: Skipping sample due to failed prior exam processing:",
                    sample["exam1"]["exam"],
                )
                return None
            item["exam"] = item["exam"] + "_" + sample["exam1"]["exam"]
            num_nodules1 = item1["num_nodules"]

            item["x"] = torch.stack([item["x"], item1["x"]])
            item["logit"] = torch.stack([item["logit"], item1["logit"]])
            item["num_nodules"].append(num_nodules1)
            item["nodule_batch_id"].extend([1] * self.args.max_num_nodules_per_scan)
            item["nodule_tp_id"].extend(
                [sample["exam1"]["screen_timepoint"]]
                * self.args.max_num_nodules_per_scan
            )

            if not self.args.precomputed_pillar_hiddens:
                for key in ["mask", "lung"]:
                    if key in item1:
                        item[key] = torch.stack([item[key], item1[key]])
                if "has_mask" in item1:
                    item["has_mask"] = [item["has_mask"], item1["has_mask"]]

            item["nodule_x"] = torch.cat([item["nodule_x"], item1["nodule_x"]], dim=0)
            for key in [
                "nodule_volumes",
                "nodule_cancer_labels",
                "nodule_luna25_labels",
                "nodule_has_cancer_labels",
                "nodule_has_luna25_labels",
                "nodule_confidence",
                "old_nodule_ids",
            ]:
                item[key] = torch.cat([item[key], item1[key]], dim=-1)

        else:
            item["num_nodules"].append(0)
            item["nodule_batch_id"].extend([1] * self.args.max_num_nodules_per_scan)
            item["nodule_tp_id"].extend(
                [item["nodule_tp_id"][0]] * self.args.max_num_nodules_per_scan
            )
            item["nodule_x"] = torch.cat(
                [item["nodule_x"], torch.zeros_like(item["nodule_x"])], 0
            )
            item["old_nodule_ids"] = torch.cat(
                [item["old_nodule_ids"], -torch.ones_like(item["old_nodule_ids"])],
                dim=-1,
            )
            for key in [
                "x",
                "logit",
            ]:
                item[key] = torch.stack([item[key], torch.zeros_like(item[key])])
            for key in [
                "nodule_cancer_labels",
                "nodule_luna25_labels",
                "nodule_has_cancer_labels",
                "nodule_has_luna25_labels",
                "nodule_volumes",
                "nodule_confidence",
            ]:
                item[key] = torch.cat([item[key], torch.zeros_like(item[key])], dim=-1)

        # Build nodule_ids_tracked: nodules with the same tracked ID across exams get the same index
        # Convert old_nodule_ids tensor to list for proper iteration
        old_nodule_ids_list = item["old_nodule_ids"].tolist()

        unique_nodules = set()
        for tp, nid in zip(item["nodule_tp_id"], old_nodule_ids_list):
            # nid is the nodid in the segmentation
            # map to tracked nodule ID if it exists
            tracked_id = (
                self.pid2tracked_nodules.get(pid, {}).get(tp, {}).get(nid, None)
            )
            if tracked_id is not None:
                unique_nodules.add(tracked_id)
            else:
                unique_nodules.add((tp, nid))  # use (tp, nid) as unique ID

        unique_nodules = list(unique_nodules)
        nodule_ids_tracked = []  # in the order of old_nodule_ids_list
        for tp, nid in zip(item["nodule_tp_id"], old_nodule_ids_list):
            tracked_id = (
                self.pid2tracked_nodules.get(pid, {}).get(tp, {}).get(nid, None)
            )
            if tracked_id is not None:
                nodule_ids_tracked.append(unique_nodules.index(tracked_id))
            else:
                nodule_ids_tracked.append(unique_nodules.index((tp, nid)))

        item["nodule_ids_tracked"] = torch.tensor(nodule_ids_tracked, dtype=torch.int64)
        for k, v in item.items():
            if isinstance(v, list):
                item[k] = torch.tensor(v)

        # take relative timepoint IDs
        item["nodule_tp_id"] = item["nodule_tp_id"] - item["nodule_tp_id"].min()

        return item

    def process_single_exam(self, sample):
        try:
            image = self.load_image(sample)
            if image is None:
                return

            resized = self.zoom_image(image, target_size=(1024, 1024))  # hard code

            if not self.args.use_pillar_processing:
                # window first
                image = self.process_image_default(image)

            # apply augmentations
            image_dict = {
                "image": image,
            }
            image_dict = self.augmentations(image_dict)

            if self.args.use_pillar_processing:
                # augmentations first: spacing, pad/crop
                # clip, apply all windows + minmax
                image = self.process_image_for_pillar(image_dict["image"])

            # load the segmentation
            segmentation_metadata = self.load_segmentation(sample)
            (
                segmentation,
                segmentation_,
                nodule_volumes,
                nodule_cancer_labels,
                nodule_luna25_labels,
                nodule_has_cancer_labels,
                nodule_has_luna25_labels,
                nodule_ids,
                num_nodules,
            ) = segmentation_metadata

            label = torch.zeros(
                image_dict["image"].shape[-3:],
                dtype=image_dict["image"].dtype,
                device=image_dict["image"].device,
            ).unsqueeze(0)

            (
                nodule_ids,
                label,
                new_nodule_indices,
                old_nodule_ids,
                nodule_volumes,
                nodule_cancer_labels,
                nodule_luna25_labels,
                nodule_has_cancer_labels,
                nodule_has_luna25_labels,
                num_nodules,
                nodule_confidence,
            ) = self.process_nodules(sample, label, segmentation_metadata)

            h1, w1, d1 = 128, 128, 32
            nodule_xs = []
            for nid in old_nodule_ids:
                ys, xs, zs = (
                    segmentation_metadata[0]["sparse_segmentation"]
                    .coalesce()
                    .indices()[
                        :,
                        segmentation_metadata[0]["sparse_segmentation"]
                        .coalesce()
                        .values()
                        == nid,
                    ]
                )
                box = {
                    "y_start": int(ys.min()),
                    "y_stop": int(ys.max()) + 1,
                    "x_start": int(xs.min()),
                    "x_stop": int(xs.max()) + 1,
                    "z_start": int(zs.min()),
                    "z_stop": int(zs.max()) + 1,
                }
                box_sample = self.patch_augmentations(
                    {
                        "image": resized,
                    }
                )
                cbbox = random_pad_3d_box(
                    box,
                    resized,
                    min_height=h1,
                    min_width=w1,
                    min_depth=d1,
                    random_hw=True,
                    random_d=True,
                )
                patch = {
                    "image": box_sample["image"][0][cbbox][None],
                }
                patch = self.patch_augmentations._transform(patch)
                nodule_xs.append(patch["image"].permute(0, 3, 1, 2))
            nodule_x = (
                torch.cat(nodule_xs, dim=0)
                if len(nodule_xs) > 0
                else torch.empty(1, d1, h1, w1)
            )

            label = label.permute(0, 3, 1, 2)
            x = image_dict["image"].permute(0, 3, 1, 2)
            try:
                mask = label.as_tensor().contiguous().to(torch.int64)
            except:
                mask = label.contiguous().to(torch.int64)

            item = {
                "x": x,
                "mask": mask,
                "y": sample["y"],
                "has_y": 1,
                "has_mask": int(sample["segmentation_path"] is not None),
                "cancer_laterality": sample["cancer_laterality"],
                "has_laterality_class": 1,
                "exam": sample["exam"],
                "pid": sample["pid"],
                "dataset": "nlst",
                "num_nodules": num_nodules,
                "nodule_ids": nodule_ids,
                "old_nodule_ids": old_nodule_ids,
                "nodule_has_cancer_labels": nodule_has_cancer_labels,
                "nodule_has_luna25_labels": nodule_has_luna25_labels,
                "nodule_volumes": nodule_volumes,
                "nodule_cancer_labels": nodule_cancer_labels,
                "nodule_luna25_labels": nodule_luna25_labels,
                "time_at_event": sample["time_at_event"],
                "y_seq": sample["y_seq"],
                "y_mask": sample["y_mask"],
                "anatomy": "chest_ct",
                "nodule_confidence": nodule_confidence,
                "nodule_x": nodule_x,
                "logit": torch.zeros(self.args.max_followup),
            }

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

    def process_single_cached_exam(self, sample):
        def pad_tensor(tensor, target_shape, value):
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor)
            tensor = torch.cat(
                [
                    tensor,
                    torch.full((target_shape,), value, dtype=torch.int64),
                ],
                dim=0,
            )
            return tensor

        try:
            if self.split_group == "test":
                sybil15_dir = "/path/to/local/cache/lung_ct/sybil15_nlst_test_hiddens/"
            else:
                sybil15_dir = "/path/to/local/cache/lung_ct/sybil15_nlst_hiddens_rve2/"
            image_embedding = torch.load(
                f"{sybil15_dir}/sample_{sample['exam']}.predictions",
                weights_only=False,
            )
            x = image_embedding["hidden"]
            logit = image_embedding["logit"]
            # load the segmentation
            segmentation_metadata = self.load_segmentation(sample)
            (
                segmentation,
                segmentation_,
                nodule_volumes,
                nodule_cancer_labels,
                nodule_luna25_labels,
                nodule_has_cancer_labels,
                nodule_has_luna25_labels,
                nodule_ids,
                num_nodules,
            ) = segmentation_metadata

            label = torch.zeros((256, 256, 256)).unsqueeze(0)
            (
                nodule_ids,
                label,
                new_nodule_indices,
                old_nodule_ids,
                nodule_volumes,
                nodule_cancer_labels,
                nodule_luna25_labels,
                nodule_has_cancer_labels,
                nodule_has_luna25_labels,
                num_nodules,
                nodule_confidence,
            ) = self.process_nodules(sample, label, segmentation_metadata)

            # sort nodules by confidence
            nodule_xs = []
            if num_nodules > 0:
                indices_by_confidence = torch.argsort(
                    nodule_confidence, descending=True
                )
                old_nodule_ids = old_nodule_ids[indices_by_confidence]
                nodule_volumes = nodule_volumes[indices_by_confidence]
                nodule_cancer_labels = nodule_cancer_labels[indices_by_confidence]
                nodule_luna25_labels = nodule_luna25_labels[indices_by_confidence]
                nodule_has_cancer_labels = nodule_has_cancer_labels[
                    indices_by_confidence
                ]
                nodule_has_luna25_labels = nodule_has_luna25_labels[
                    indices_by_confidence
                ]
                nodule_confidence = nodule_confidence[indices_by_confidence]

                nodules_found = []
                for i, nid in enumerate(old_nodule_ids):
                    if not os.path.exists(
                        f"/path/to/local/cache/lung_ct/nlst_abnormalities51_test_patches/{sample['exam']}_nodule{nid}.pt"
                    ):
                        # print(f"Missing patch for {sample['exam']}_nodule{nid}")
                        # print("OLD IDS", old_nodule_ids)
                        # print("INDICES", indices_by_confidence)
                        nodules_found.append(False)
                        continue

                    patch = torch.load(
                        f"/path/to/local/cache/lung_ct/nlst_abnormalities51_test_patches/{sample['exam']}_nodule{nid}.pt",
                        weights_only=False,
                    )
                    patch = self.process_nodule_patch(patch)
                    nodule_xs.append(patch if patch.dim() == 4 else patch.unsqueeze(0))
                    nodules_found.append(True)

                # filter out nodules without patches
                nodules_found = torch.tensor(nodules_found, dtype=torch.bool)
                old_nodule_ids = old_nodule_ids[nodules_found]
                nodule_volumes = nodule_volumes[nodules_found]
                nodule_cancer_labels = nodule_cancer_labels[nodules_found]
                nodule_luna25_labels = nodule_luna25_labels[nodules_found]
                nodule_has_cancer_labels = nodule_has_cancer_labels[nodules_found]
                nodule_has_luna25_labels = nodule_has_luna25_labels[nodules_found]
                nodule_confidence = nodule_confidence[nodules_found]

            if len(nodule_xs) < self.args.max_num_nodules_per_scan:
                pad_length = self.args.max_num_nodules_per_scan - num_nodules
                nodule_xs.append(
                    torch.zeros(
                        (
                            pad_length,
                            32,
                            128,
                            128,
                        ),
                        dtype=torch.float32,
                    )
                )
                old_nodule_ids = pad_tensor(old_nodule_ids, pad_length, -1)
                nodule_volumes = pad_tensor(nodule_volumes, pad_length, 0)
                nodule_cancer_labels = pad_tensor(nodule_cancer_labels, pad_length, -1)
                nodule_luna25_labels = pad_tensor(nodule_luna25_labels, pad_length, -1)
                nodule_has_cancer_labels = pad_tensor(
                    nodule_has_cancer_labels, pad_length, 0
                )
                nodule_has_luna25_labels = pad_tensor(
                    nodule_has_luna25_labels, pad_length, 0
                )
                nodule_confidence = pad_tensor(nodule_confidence, pad_length, 0.0)

            else:
                nodule_xs = nodule_xs[: self.args.max_num_nodules_per_scan]
                old_nodule_ids = old_nodule_ids[: self.args.max_num_nodules_per_scan]
                nodule_volumes = nodule_volumes[: self.args.max_num_nodules_per_scan]
                nodule_cancer_labels = nodule_cancer_labels[
                    : self.args.max_num_nodules_per_scan
                ]
                nodule_luna25_labels = nodule_luna25_labels[
                    : self.args.max_num_nodules_per_scan
                ]
                nodule_has_cancer_labels = nodule_has_cancer_labels[
                    : self.args.max_num_nodules_per_scan
                ]
                nodule_has_luna25_labels = nodule_has_luna25_labels[
                    : self.args.max_num_nodules_per_scan
                ]
                nodule_confidence = nodule_confidence[
                    : self.args.max_num_nodules_per_scan
                ]

            nodule_x = torch.cat(nodule_xs, dim=0) if len(nodule_xs) > 0 else None

            item = {
                "x": x,
                "logit": logit,
                "y": sample["y"],
                "has_y": 1,
                "has_mask": int(sample["segmentation_path"] is not None),
                "cancer_laterality": sample["cancer_laterality"],
                "has_laterality_class": 1,
                "exam": sample["exam"],
                "pid": sample["pid"],
                "dataset": "nlst",
                "num_nodules": num_nodules,
                "nodule_ids": nodule_ids,
                "old_nodule_ids": old_nodule_ids,
                "nodule_has_cancer_labels": nodule_has_cancer_labels,
                "nodule_has_luna25_labels": nodule_has_luna25_labels,
                "nodule_volumes": nodule_volumes,
                "nodule_cancer_labels": nodule_cancer_labels,
                "nodule_luna25_labels": nodule_luna25_labels,
                "time_at_event": sample["time_at_event"],
                "y_seq": sample["y_seq"],
                "y_mask": sample["y_mask"],
                "anatomy": "chest_ct",
                "nodule_confidence": nodule_confidence,
                "nodule_x": nodule_x,
            }

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

    def skip_sample(self, series_id, series_dict, exam_dict, pt_metadata):
        if super().skip_sample(series_id, series_dict, exam_dict, pt_metadata):
            return True

        examid = get_examid(
            pt_metadata["pid"][0], exam_dict["screen_timepoint"], series_id
        )
        if (self.split_group != "test") and (
            examid not in self.annotations_metadata["pillar_exams"]
        ):
            return True
        return False

    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed NLST CT Cancer Risk {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
        class_balance = Counter([d["exam2"]["y"] for d in dataset])
        exams = set(
            [d["exam2"]["exam"] for d in dataset]
            + [d["exam1"]["exam"] for d in dataset if d["exam1"]]
        )
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group, len(dataset), len(exams), len(patients), class_balance
        )
        statement += "\n" + "Censor Times: {}".format(
            Counter([d["exam2"]["time_at_event"] for d in dataset])
        )
        statement
        return statement

    def load_segmentation(self, sample):
        if sample["segmentation_path"] is not None:
            segmentation = torch.load(sample["segmentation_path"])
            sparse_segmentation = segmentation["sparse_segmentation"]
            nodule_volumes = segmentation["nodule_volumes"]
            nodule_ids = segmentation["nodule_ids"]
            if "nodule_cancer_labels" in segmentation:
                nodule_cancer_labels = segmentation["nodule_cancer_labels"]
                nodule_luna25_labels = segmentation["nodule_luna25_labels"]
                nodule_has_cancer_labels = segmentation["nodule_has_cancer_labels"]
                nodule_has_luna25_labels = segmentation["nodule_has_luna25_labels"]
            else:
                nodule_cancer_labels = torch.tensor([0 for _ in range(len(nodule_ids))])
                nodule_luna25_labels = torch.tensor([0 for _ in range(len(nodule_ids))])
                nodule_has_cancer_labels = torch.tensor(
                    [0 for _ in range(len(nodule_ids))]
                )
                nodule_has_luna25_labels = torch.tensor(
                    [0 for _ in range(len(nodule_ids))]
                )

            num_nodules = len(nodule_ids)
            segmentation_ = sparse_segmentation.to_dense()

            # resize segmentation_
            segmentation_ = segmentation_.permute(2, 0, 1).unsqueeze(1)
            segmentation_ = F.interpolate(
                segmentation_,
                size=(self.args.img_size[0], self.args.img_size[1]),
                mode="nearest-exact",
            )
            # Remove batch/channel dims and permute back: (H, W, D)
            segmentation_ = segmentation_.squeeze(1).permute(1, 2, 0)
        else:
            segmentation = segmentation_ = None
            nodule_volumes = torch.tensor([0])
            nodule_cancer_labels = torch.tensor([0])
            nodule_luna25_labels = torch.tensor([0])
            nodule_has_cancer_labels = torch.tensor([0])
            nodule_has_luna25_labels = torch.tensor([0])
            nodule_ids = torch.tensor([0])
            num_nodules = 0

        return (
            segmentation,
            segmentation_,
            nodule_volumes,
            nodule_cancer_labels,
            nodule_luna25_labels,
            nodule_has_cancer_labels,
            nodule_has_luna25_labels,
            nodule_ids,
            num_nodules,
        )

    def process_nodule_patch(self, patch):
        patch = torch.from_numpy(patch).float()
        patch = self.process_image_default(patch).unsqueeze(0)
        patch = patch.permute(0, 3, 1, 2)  # B, D, H, W
        patch = F.interpolate(
            patch,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)  # B, H, W, D

        image_dict = {
            "image": patch,
        }
        image_dict = self.patch_augmentations._transform(image_dict)
        image_dict = self.patch_augmentations.predict_transforms.transforms[-1](
            image_dict
        )
        patch = image_dict["image"].permute(0, 3, 1, 2)
        return patch

class NLSTSybilTest(NLST_Longitudinal):
    test_dataset = pickle.load(
        open(
            "/data/rbg/users/pgmikhael/current/notebooks/LungCT/Sybil/ResultsDir/nlst_dataset.p",
            "rb",
        )
    )["test"]
    test_exams = set(str(sample["exam"]) for sample in test_dataset)

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

        pid_tracked_nodules = pickle.load(
            open(
                "/data/rbg/users/pgmikhael/current/Sybil/notebooks/NLSTNodules/pid2tracked_nodules.p",
                "rb",
            )
        )
        # pid_tracked_nodules is an iterable of (pid, nid2nodes)
        # Build pid2tracked_exams and pid2tracked_nodules ensuring nested dicts
        self.pid2tracked_nodules = {}
        pid2tracked_exams = {}
        for pid, nid2nodes in pid_tracked_nodules:
            # initialize the set of tracked exams for this pid
            pid2tracked_exams.setdefault(pid, set())
            for nid, tps in nid2nodes.items():
                for tp, node in tps.items():
                    pid2tracked_exams[pid].add(node["exam"])
                    # ensure nested dicts exist so we can index to depth 3 safely
                    if pid not in self.pid2tracked_nodules:
                        self.pid2tracked_nodules[pid] = {}
                    if tp not in self.pid2tracked_nodules[pid]:
                        self.pid2tracked_nodules[pid][tp] = {}
                    self.pid2tracked_nodules[pid][tp][node["nodid_in_segmentation"]] = (
                        nid
                    )

        dataset = []
        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row.get("split", "train"),
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            if (split != split_group) and not self.args.turn_off_splits:
                continue

            exams_by_timepoint = {}

            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():
                    series_dict["series_data"] = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in series_dict["series_data"].items()
                    }
                    if self.skip_sample(series_id, series_dict, exam_dict, pt_metadata):
                        continue

                    sample = self.get_volume_dict(
                        series_id,
                        series_dict,
                        exam_dict,
                        pt_metadata,
                        pid,
                        split,
                    )
                    if len(sample) == 0:
                        continue

                    timepoint = sample["screen_timepoint"]
                    examid = sample["exam"]
                    exams_by_timepoint[timepoint] = sample

            for tp1, tp2 in [(0, 1), (1, 2), (0, 2)]:
                if (tp1 in exams_by_timepoint) and (tp2 in exams_by_timepoint):
                    sample = {
                        "pid": pid,
                        "exam1": exams_by_timepoint[tp1],
                        "exam2": exams_by_timepoint[tp2],
                        "y": exams_by_timepoint[tp2]["y"],
                        "time_at_event": exams_by_timepoint[tp2]["time_at_event"],
                    }
                    
                    dataset.append(sample)
            if len(exams_by_timepoint) == 1:
                tp = list(exams_by_timepoint.keys())[0]
                sample = {
                    "pid": pid,
                    "exam1": None,
                    "exam2": exams_by_timepoint[tp],
                    "y": exams_by_timepoint[tp]["y"],
                    "time_at_event": exams_by_timepoint[tp]["time_at_event"],
                }
                
                dataset.append(sample)

        return dataset

    def skip_sample(self, series_id, series_dict, exam_dict, pt_metadata):
        exam3s = "{}{}{}".format(
            pt_metadata["pid"][0],
            exam_dict["screen_timepoint"],
            int(series_id.split(".")[-1][-3:]),
        )
        if exam3s not in self.test_exams:
            return True

        examid = get_examid(
            pt_metadata["pid"][0], exam_dict["screen_timepoint"], series_id
        )
        """ 
        all_exams = [
            get_examid(pt_metadata["pid"][0], exam_dict["screen_timepoint"], sid)
            for sid, sdict in exam_dict["image_series"].items()
            if not self.is_localizer(sdict["series_data"])
        ]
        annotated_exams = [
            examid
            for examid in all_exams
            if examid in self.annotations_metadata["exams"]
        ]

        if len(annotated_exams) > 0 and (examid not in annotated_exams):
            return True
        """

        return False

