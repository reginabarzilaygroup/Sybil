import numpy as np
from tqdm import tqdm
from ast import literal_eval
from sybil.datasets.nlst import NLST_Survival_Dataset
from collections import Counter
import copy

DEVICE_ID = {
    "GE MEDICAL SYSTEMS": 0,
    "TOSHIBA": 1,
    "Philips": 2,
    "SIEMENS": 3,
    "Siemens Healthcare": 3,  # note: same id as SIEMENS
    "Vital Images, Inc.": 4,
    "Hitachi Medical Corporation": 5,
    "LightSpeed16": 6,
}


class MGH_Dataset(NLST_Survival_Dataset):
    """
    MGH Dataset Cohort 1
    """

    def create_dataset(self, split_group):
        """
        Gets the dataset from the paths and labels in the json.
        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
        Returns:
            The dataset as a dictionary with img paths, label,
            and additional information regarding exam or participant
        """
        dataset = []

        # if split probs is set, randomly assign new splits, (otherwise default is 70% train, 15% dev and 15% test)
        if self.args.assign_splits:
            np.random.seed(self.args.cross_val_seed)
            self.assign_splits(self.metadata_json)

        for mrn_row in tqdm(self.metadata_json):
            pid, split, exams = mrn_row["pid"], mrn_row["split"], mrn_row["accessions"]
            # pt_metadata missing

            for exam_dict in exams:
                studyuid = exam_dict["StudyInstanceUID"]
                bridge_uid = exam_dict["bridge_uid"]
                days_to_last_exam = -int(
                    exam_dict["diff_days"]
                )  # no. of days to the oldest exam (0 or a negative int)

                exam_no = self.get_exam_no(days_to_last_exam, exams)

                y, y_seq, y_mask, time_at_event = self.get_label(exam_dict, exams)

                for series_id, series_dict in exam_dict["image_series"].items():

                    if self.skip_sample(series_dict, exam_dict, mrn_row, split_group):
                        continue

                    img_paths = series_dict["paths"]
                    img_paths = [p.replace("Data082021", "pngs") for p in img_paths]
                    slice_locations = series_dict["image_posn"]
                    series_data = series_dict["series_data"]
                    device = DEVICE_ID[series_data["Manufacturer"]]

                    sorted_img_paths, sorted_slice_locs = self.order_slices(
                        img_paths, slice_locations
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
                        "exam_str": "{}_{}".format(bridge_uid, exam_no),
                        "accession": exam_no,
                        "study": studyuid,
                        "series": series_id,
                        "pid": pid,
                        "device": device,
                        "lung_rads": -1
                        if exam_dict["lung_rads"] == np.nan
                        else exam_dict["lung_rads"],
                        "IV_contrast": exam_dict["IV_contrast"],
                        "lung_cancer_screening": exam_dict["lung_cancer_screening"],
                        "cancer_location": np.zeros(14),  # mgh has no annotations
                        "cancer_laterality": np.zeros(
                            3, dtype=np.int
                        ),  # has to be int, while cancer_location has to be float
                        "num_original_slices": len(series_dict["paths"]),
                        "annotations": [],
                        "pixel_spacing": series_dict["pixel_spacing"]
                        + [series_dict["slice_thickness"]],
                        "slice_thickness": self.get_slice_thickness_class(
                            series_dict["slice_thickness"]
                        ),
                    }

                    if self.args.use_risk_factors:
                        sample["risk_factors"] = self.get_risk_factors(
                            exam_dict, return_dict=False
                        )

                    if self.args.use_annotations:
                        # mgh has no annotations, so set everything to zero / false
                        sample["volume_annotations"] = np.array(
                            [0 for _ in sample["paths"]]
                        )
                        sample["annotations"] = [
                            {"image_annotations": None} for path in sample["paths"]
                        ]

                    dataset.append(sample)

        return dataset

    def skip_sample(self, series_dict, exam_dict, mrn_row, split):
        if not mrn_row["split"] == split:
            return True

        if mrn_row["in_cohort2"]:
            return True

        # check if screen is localizer screen or not enough images
        if self.is_localizer(series_dict["series_data"]):
            return True

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

        # remove where slice location doesn't change (different axis):
        if len(set(series_dict["image_posn"])) < 2:
            return True

        if len(series_dict["paths"]) < self.args.min_num_images:
            return True

        return False

    def get_exam_no(self, diff_days, exams):
        """Gets the index of the exam, compared to the other exams"""
        sorted_days = sorted([-exam["diff_days"] for exam in exams], reverse=True)
        return sorted_days.index(diff_days)

    def get_label(self, exam_dict, exams):
        is_cancer_cohort = exam_dict["cancer_cohort_yes_no"] == "yes"
        days_to_last_followup = -exam_dict["diff_days"]
        years_to_last_followup = days_to_last_followup // 365

        y = 0
        y_seq = np.zeros(self.args.max_followup)
        if is_cancer_cohort:
            days_to_cancer = -exam_dict["diff_days_exam_lung_cancer_diagnosis"]
            years_to_cancer = int(days_to_cancer // 365)
            y = years_to_cancer < self.args.max_followup

            time_at_event = min(years_to_cancer, self.args.max_followup - 1)
            y_seq[years_to_cancer:] = 1
        else:
            time_at_event = min(years_to_last_followup, self.args.max_followup - 1)

        y_mask = np.array(
            [1] * (time_at_event + 1)
            + [0] * (self.args.max_followup - (time_at_event + 1))
        )
        y_mask = y_mask[: self.args.max_followup]
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

    def get_risk_factors(self, exam_dict, return_dict=False):
        risk_factors = {
            "age_at_exam": exam_dict["age_at_exam"],
            "pack_years": exam_dict["pack_years"],
            "race": exam_dict["race"],
            "sex": exam_dict["sex"],
            "smoking_status": exam_dict["smoking_status"],
        }

        if return_dict:
            return risk_factors
        else:
            return np.array(
                [v for v in risk_factors.values() if not isinstance(v, str)]
            )

    def is_localizer(self, series_dict):
        is_localizer = "LOCALIZER" in literal_eval(series_dict["ImageType"])
        return is_localizer

    @staticmethod
    def set_args(args):
        args.num_classes = args.max_followup

    def get_summary_statement(self, dataset, split_group):
        summary = "Constructed MGH CT Cancer Survival {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
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


class MGH_Screening(NLST_Survival_Dataset):
    """
    MGH Dataset Cohort 2
    """

    def create_dataset(self, split_group):
        """
        Gets the dataset from the paths and labels in the json.
        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
        Returns:
            The dataset as a dictionary with img paths, label,
            and additional information regarding exam or participant
        """
        assert not self.args.train, "Cohort 2 should not be used for training"

        dataset = []

        for mrn_row in tqdm(self.metadata_json):
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

                    dataset.append(sample)

        return dataset

    def skip_sample(self, series_dict, exam_dict, mrn_row):
        # unknown cancer status
        if exam_dict["Future_cancer"] == "unkown":
            return True

        if (exam_dict["days_before_cancer_dx"] < 0) or (
            exam_dict["days_to_last_follow_up"] < 0
        ):
            return True

        # check if screen is localizer screen or not enough images
        if self.is_localizer(series_dict["series_data"]):
            return True

        slice_thickness = series_dict["SliceThickness"]
        # check if restricting to specific slice thicknesses
        if (self.args.slice_thickness_filter is not None) and (
            (slice_thickness in ["", None])
            or (slice_thickness > self.args.slice_thickness_filter)
            or (slice_thickness < 0)
        ):
            return True

        if series_dict["PixelSpacing"] is None:
            return True

        if len(series_dict["paths"]) < self.args.min_num_images:
            return True

        return False

    def get_volume_dict(self, series_id, series_dict, exam_dict, mrn_row):

        img_paths = series_dict["paths"]
        img_paths = [
            p.replace("MIT_Lung_Cancer_Screening", "screening_pngs").replace(
                ".dcm", ".png"
            )
            for p in img_paths
        ]
        slice_locations = series_dict["slice_location"]
        series_data = series_dict["series_data"]
        pixel_spacing = series_dict["PixelSpacing"] + [series_dict["SliceThickness"]]
        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations, reverse=True
        )

        device = DEVICE_ID[series_data["Manufacturer"]]

        studyuid = exam_dict["StudyInstanceUID"]
        bridge_uid = exam_dict["bridge_uid"]

        y, y_seq, y_mask, time_at_event = self.get_label(exam_dict, mrn_row)

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
            "bridge_uid": bridge_uid,
            "device": device,
            "lung_rads": exam_dict["LR Score"],
            "cancer_location": np.zeros(14),  # mgh has no annotations
            "cancer_laterality": np.zeros(
                3, dtype=np.int
            ),  # has to be int, while cancer_location has to be float
            "num_original_slices": len(series_dict["paths"]),
            "marital_status": exam_dict["marital_status"],
            "religion": exam_dict["religion"],
            "primary_site": exam_dict["Primary Site"],
            "laterality1": exam_dict["Laterality"],
            "laterality2": exam_dict["Laterality.1"],
            "icdo3": exam_dict["Histo/Behavior ICD-O-3"],
            "pixel_spacing": pixel_spacing,
            "slice_thickness": self.get_slice_thickness_class(pixel_spacing[-1]),
        }

        if self.args.use_risk_factors:
            sample["risk_factors"] = self.get_risk_factors(exam_dict, return_dict=False)

        if self.args.use_annotations:
            # mgh has no annotations, so set everything to zero / false
            sample["volume_annotations"] = np.array([0 for _ in sample["paths"]])
            sample["annotations"] = [
                {"image_annotations": None} for path in sample["paths"]
            ]
        return sample

    def get_label(self, exam_dict, mrn_row):
        is_cancer_cohort = exam_dict["Future_cancer"].lower().strip() == "yes"
        days_to_cancer = exam_dict["days_before_cancer_dx"]

        y = False
        if (
            is_cancer_cohort
            and (not np.isnan(days_to_cancer))
            and (days_to_cancer > -1)
        ):
            years_to_cancer = int(days_to_cancer // 365)
            y = years_to_cancer < self.args.max_followup

        y_seq = np.zeros(self.args.max_followup)

        if y:
            time_at_event = years_to_cancer
            y_seq[years_to_cancer:] = 1
        else:
            if is_cancer_cohort:
                assert (days_to_cancer < 0) or (
                    years_to_cancer >= self.args.max_followup
                )
                time_at_event = self.args.max_followup - 1
            else:
                days_to_last_neg_followup = exam_dict["days_to_last_follow_up"]
                years_to_last_neg_followup = int(days_to_last_neg_followup // 365)
                time_at_event = min(
                    years_to_last_neg_followup, self.args.max_followup - 1
                )

        y_mask = np.array(
            [1] * (time_at_event + 1)
            + [0] * (self.args.max_followup - (time_at_event + 1))
        )
        y_mask = y_mask[: self.args.max_followup]
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

    def get_risk_factors(self, exam_dict, return_dict=False):
        risk_factors = {
            "race": exam_dict["race"],
            "pack_years": exam_dict["Packs Years"],
            "age_at_exam": exam_dict["age at the exam"],
            "gender": exam_dict["gender"],
            "smoking_status": exam_dict["Smoking Status"],
            "lung_rads": exam_dict["LR Score"],
            "years_since_quit_smoking": exam_dict["Year Since Last Smoked"],
        }

        if return_dict:
            return risk_factors
        else:
            return np.array(
                [v for v in risk_factors.values() if not isinstance(v, str)]
            )

    def is_localizer(self, series_dict):
        is_localizer = "LOCALIZER" in literal_eval(series_dict["ImageType"])
        return is_localizer

    @staticmethod
    def set_args(args):
        args.num_classes = args.max_followup

    def get_summary_statement(self, dataset, split_group):
        summary = "Constructed MGH CT Cancer Survival {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
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
