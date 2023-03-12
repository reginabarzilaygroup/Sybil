import os
from posixpath import split
import traceback, warnings
import pickle, json
import numpy as np
import pydicom
import torchio as tio
from tqdm import tqdm
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils import data
from sybil.serie import Serie
from sybil.utils.loading import get_sample_loader
from sybil.datasets.utils import (
    METAFILE_NOTFOUND_ERR,
    LOAD_FAIL_MSG,
    VOXEL_SPACING,
)
import copy
from sybil.datasets.nlst_risk_factors import NLSTRiskFactorVectorizer

METADATA_FILENAME = {"google_test": "NLST/full_nlst_google.json"}

GOOGLE_SPLITS_FILENAME = (
    "/Mounts/rbg-storage1/datasets/NLST/Shetty_et_al(Google)/data_splits.p"
)

CORRUPTED_PATHS = "/Mounts/rbg-storage1/datasets/NLST/corrupted_img_paths.pkl"

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
        self.always_resample_pixel_spacing = split_group in ["dev", "test"]

        self.resample_transform = tio.transforms.Resample(target=VOXEL_SPACING)
        self.padding_transform = tio.transforms.CropOrPad(
            target_shape=tuple(args.img_size + [args.num_images]), padding_mode=0
        )

        if args.use_annotations:
            assert (
                self.args.region_annotations_filepath
            ), "ANNOTATIONS METADATA FILE NOT SPECIFIED"
            self.annotations_metadata = json.load(
                open(self.args.region_annotations_filepath, "r")
            )

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        print(self.get_summary_statement(self.dataset, split_group))

        dist_key = "y"
        label_dist = [d[dist_key] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1.0 / len(label_counts)
        label_weights = {
            label: weight_per_label / count for label, count in label_counts.items()
        }

        print("Class counts are: {}".format(label_counts))
        print("Label weights are {}".format(label_weights))
        self.weights = [label_weights[d[dist_key]] for d in self.dataset]

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
        # self.risk_factor_vectorizer = NLSTRiskFactorVectorizer(self.args)

        if self.args.assign_splits:
            np.random.seed(self.args.cross_val_seed)
            self.assign_splits(self.metadata_json)

        dataset = []

        for mrn_row in tqdm(self.metadata_json, position=0):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["split"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            if not split == split_group:
                continue

            for exam_dict in exams:

                if self.args.use_only_thin_cuts_for_ct and split_group in [
                    "train",
                    "dev",
                ]:
                    thinnest_series_id = self.get_thinnest_cut(exam_dict)

                elif split == "test" and self.args.assign_splits:
                    thinnest_series_id = self.get_thinnest_cut(exam_dict)

                elif split == "test":
                    google_series = list(self.GOOGLE_SPLITS[pid]["exams"])
                    nlst_series = list(exam_dict["image_series"].keys())
                    thinnest_series_id = [s for s in nlst_series if s in google_series]
                    assert len(thinnest_series_id) < 2
                    if len(thinnest_series_id) > 0:
                        thinnest_series_id = thinnest_series_id[0]
                    elif len(thinnest_series_id) == 0:
                        if self.args.assign_splits:
                            thinnest_series_id = self.get_thinnest_cut(exam_dict)
                        else:
                            continue

                for series_id, series_dict in exam_dict["image_series"].items():
                    if self.skip_sample(series_dict, pt_metadata):
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

    def skip_sample(self, series_dict, pt_metadata):
        series_data = series_dict["series_data"]
        # check if screen is localizer screen or not enough images
        is_localizer = self.is_localizer(series_data)

        # check if restricting to specific slice thicknesses
        slice_thickness = series_data["reconthickness"][0]
        wrong_thickness = (self.args.slice_thickness_filter is not None) and (
            slice_thickness not in self.args.slice_thickness_filter
        )

        # check if valid label (info is not missing)
        screen_timepoint = series_data["study_yr"][0]
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
        device = series_data["manufacturer"][0]
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

        y, y_seq, y_mask, time_at_event = self.get_label(pt_metadata, screen_timepoint)

        exam_int = int(
            "{}{}{}".format(
                int(pid), int(screen_timepoint), int(series_id.split(".")[-1][-3:])
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

        return sample

    def check_label(self, pt_metadata, screen_timepoint):
        valid_days_since_rand = (
            pt_metadata["scr_days{}".format(screen_timepoint)][0] > -1
        )
        valid_days_to_cancer = pt_metadata["candx_days"][0] > -1
        valid_followup = pt_metadata["fup_days"][0] > -1
        return (valid_days_since_rand) and (valid_days_to_cancer or valid_followup)

    def get_label(self, pt_metadata, screen_timepoint):
        days_since_rand = pt_metadata["scr_days{}".format(screen_timepoint)][0]
        days_to_cancer_since_rand = pt_metadata["candx_days"][0]
        days_to_cancer = days_to_cancer_since_rand - days_since_rand
        years_to_cancer = (
            int(days_to_cancer // 365) if days_to_cancer_since_rand > -1 else 100
        )
        days_to_last_followup = int(pt_metadata["fup_days"][0] - days_since_rand)
        years_to_last_followup = days_to_last_followup // 365
        y = years_to_cancer < self.args.max_followup
        y_seq = np.zeros(self.args.max_followup)
        cancer_timepoint = pt_metadata["cancyr"][0]
        if y:
            if years_to_cancer > -1:
                assert screen_timepoint <= cancer_timepoint
            time_at_event = years_to_cancer
            y_seq[years_to_cancer:] = 1
        else:
            time_at_event = min(years_to_last_followup, self.args.max_followup - 1)
        y_mask = np.array(
            [1] * (time_at_event + 1)
            + [0] * (self.args.max_followup - (time_at_event + 1))
        )
        assert len(y_mask) == self.args.max_followup
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

    def is_localizer(self, series_dict):
        is_localizer = (
            (series_dict["imageclass"][0] == 0)
            or ("LOCALIZER" in series_dict["imagetype"][0])
            or ("TOP" in series_dict["imagetype"][0])
        )
        return is_localizer

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

    def order_slices(self, img_paths, slice_locations):
        sorted_ids = np.argsort(slice_locations)
        sorted_img_paths = np.array(img_paths)[sorted_ids].tolist()
        sorted_slice_locs = np.sort(slice_locations).tolist()

        if not sorted_img_paths[0].startswith(self.args.img_dir):
            sorted_img_paths = [
                self.args.img_dir
                + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                for path in sorted_img_paths
            ]
        if (
            self.args.img_file_type == "dicom"
        ):  # ! NOTE: removing file extension affects get_ct_annotations mapping path to annotation
            sorted_img_paths = [
                path.replace("nlst-ct-png", "nlst-ct").replace(".png", "")
                for path in sorted_img_paths
            ]

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
    def METADATA_FILENAME(self):
        return METADATA_FILENAME["google_test"]

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
            if (
                self.args.img_file_type == "dicom"
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
        sample = self.dataset[index]
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
        resample_now = self.args.resample_pixel_spacing_prob > np.random.uniform()
        if self.always_resample_pixel_spacing or resample_now:
            spacing = torch.tensor(sample["pixel_spacing"] + [1])
            input_arr = tio.ScalarImage(
                affine=torch.diag(spacing),
                tensor=input_arr.permute(0, 2, 3, 1),
            )
            input_arr = self.resample_transform(input_arr)
            input_arr = self.padding_transform(input_arr.data)

            if self.args.use_annotations:
                mask_arr = tio.ScalarImage(
                    affine=torch.diag(spacing),
                    tensor=mask_arr.permute(0, 2, 3, 1),
                )
                mask_arr = self.resample_transform(mask_arr)
                mask_arr = self.padding_transform(mask_arr.data)

        out_dict["input"] = input_arr.data.permute(0, 3, 1, 2)
        if self.args.use_annotations:
            out_dict["mask"] = mask_arr.data.permute(0, 3, 1, 2)

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


class NLST_for_PLCO(NLST_Survival_Dataset):
    """
    Dataset for risk factor-based risk model
    """

    def get_volume_dict(
        self, series_id, series_dict, exam_dict, pt_metadata, pid, split
    ):
        series_data = series_dict["series_data"]
        screen_timepoint = series_data["study_yr"][0]
        assert screen_timepoint == exam_dict["screen_timepoint"]

        y, y_seq, y_mask, time_at_event = self.get_label(pt_metadata, screen_timepoint)

        exam_int = int(
            "{}{}{}".format(
                int(pid), int(screen_timepoint), int(series_id.split(".")[-1][-3:])
            )
        )

        riskfactors = self.get_risk_factors(
            pt_metadata, screen_timepoint, return_dict=True
        )

        riskfactors["education"] = EDUCAT_LEVEL.get(riskfactors["education"], -1)
        riskfactors["race"] = RACE_ID_KEYS.get(pt_metadata["race"][0], -1)

        sample = {
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
        }
        sample.update(riskfactors)

        if (
            riskfactors["education"] == -1
            or riskfactors["race"] == -1
            or pt_metadata["weight"][0] == -1
            or pt_metadata["height"][0] == -1
        ):
            return {}

        return sample


class NLST_for_PLCO_Screening(NLST_for_PLCO):
    def create_dataset(self, split_group):
        generated_lung_rads = pickle.load(
            open("/data/rsg/mammogram/NLST/nlst_acc2lungrads.p", "rb")
        )
        dataset = super().create_dataset(split_group)
        # get lung rads for each year
        pid2lungrads = {}
        for d in dataset:
            lungrads = generated_lung_rads[d["exam"]]
            if d["pid"] in pid2lungrads:
                pid2lungrads[d["pid"]][d["screen_timepoint"]] = lungrads
            else:
                pid2lungrads[d["pid"]] = {d["screen_timepoint"]: lungrads}
        plco_results_dataset = []
        for d in dataset:
            if len(pid2lungrads[d["pid"]]) < 3:
                continue
            is_third_screen = d["screen_timepoint"] == 2
            is_1yr_ca_free = (d["y"] and d["time_at_event"] > 0) or (not d["y"])
            if is_third_screen and is_1yr_ca_free:
                d["scr_group_coef"] = self.get_screening_group(pid2lungrads[d["pid"]])
                for k in ["age", "years_since_quit_smoking", "smoking_duration"]:
                    d[k] = d[k] + 1
                plco_results_dataset.append(d)
            else:
                continue
        return plco_results_dataset

    def get_screening_group(self, lung_rads_dict):
        """doi:10.1001/jamanetworkopen.2019.0204 Table 1"""
        scr1, scr2, scr3 = lung_rads_dict[0], lung_rads_dict[1], lung_rads_dict[2]

        if all([not scr1, not scr2, not scr3]):
            return 0
        elif (not scr3) and ((not scr1) or (not scr2)):
            return 0.6554117
        elif ((not scr3) and all([scr1, scr2])) or (
            all([not scr1, not scr2]) and (scr3)
        ):
            return 0.9798233
        elif (
            (all([scr1, scr3]) and not scr2)
            or (not scr1 and all([scr2, scr3]))
            or (all([scr1, scr2, scr3]))
        ):
            return 2.1940610
        raise ValueError(
            "Screen {} has not equivalent PLCO group".format(lung_rads_dict)
        )


class NLST_Risk_Factor_Task(NLST_Survival_Dataset):
    """
    Dataset for risk factor-based risk model
    """

    def get_risk_factors(self, pt_metadata, screen_timepoint, return_dict=False):
        return self.risk_factor_vectorizer.get_risk_factors_for_sample(
            pt_metadata, screen_timepoint
        )
