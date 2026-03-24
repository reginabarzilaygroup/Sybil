import traceback, warnings
import copy
from sybil.datasets.luna import (
    LUNA_Patches,
    LUNA_Confidence_Flat,
)
from sybil.datasets.nlst import (
    NLST_Survival_Dataset,
    NLST_Patches,
    NLST_Confidence_Flat,
)
from sybil.datasets.utils import LOAD_FAIL_MSG



class NodulesPatches(NLST_Survival_Dataset):
    def __init__(self, args, split_group):
        self.args = args
        luna_args = copy.deepcopy(args)
        nlst_args = copy.deepcopy(args)
        # dataset paths
        luna_args.dataset_file_path = (
            "/data/rbg/shared/datasets/LUNA16/LUNA16/luna_dataset_v4.json"
        )
        nlst_args.dataset_file_path = (
            "/data/rbg/shared/datasets/NLST/NLST/stmix_segmentation_dataset.json"
        )

        # loaders
        luna_args.input_loader_name = "biomedparse_loader"
        nlst_args.input_loader_name = "dicom_loader"

        if args.cache_path is not None:
            luna_args.cache_path = "/data/rbg/shared/datasets/LUNA16/LUNA16/monai_cache"
            nlst_args.cache_path = (
                None  # "/data/rbg/shared/datasets/NLST/NLST/monai_cache"
            )

        # datasets
        self.luna = LUNA_Patches(luna_args, split_group)
        self.nlst = NLST_Patches(nlst_args, split_group)
        self.dataset = self.create_dataset(split_group)

    def create_dataset(self, split_group):
        luna_dataset = self.luna.dataset
        # lndb_dataset = self.lndb.dataset
        nlst_dataset = self.nlst.dataset
        # add item to indicate dataset
        for item in luna_dataset:
            item["dataset"] = "luna"

        for item in nlst_dataset:
            item["dataset"] = "nlst"

        dataset = nlst_dataset + luna_dataset

        if self.args.predict:
            # filter to only one example per vol_exam
            vol_exams = set()
            filtered_dataset = []
            for d in dataset:
                if str(d["vol_exam"]) in vol_exams:
                    continue
                vol_exams.add(str(d["vol_exam"]))
                filtered_dataset.append(d)
            dataset = filtered_dataset

        return dataset

    def __getitem__(self, index):
        sample = copy.deepcopy(self.dataset[index])
        dataset = sample["dataset"]
        try:
            if dataset == "luna":
                item = self.luna.process_item(sample)
            elif dataset == "nlst":
                item = self.nlst.process_item(sample)
            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))



class NodulesConfidenceFlat(NodulesPatches):
    def __init__(self, args, split_group):
        self.args = args
        luna_args = copy.deepcopy(args)
        nlst_args = copy.deepcopy(args)
        # dataset paths
        luna_args.dataset_file_path = "/data/rbg/shared/datasets/LUNA16/LUNA16/luna_dataset_v4_predicted_boxes.json"
        nlst_args.dataset_file_path = "/data/rbg/shared/datasets/NLST/NLST/stmix_segmentation_predicted_boxes_dataset.json"

        # loaders
        luna_args.input_loader_name = "biomedparse_loader"
        nlst_args.input_loader_name = "dicom_loader"

        if args.cache_path is not None:
            luna_args.cache_path = "/data/rbg/shared/datasets/LUNA16/LUNA16/monai_cache"
            nlst_args.cache_path = (
                None  # "/data/rbg/shared/datasets/NLST/NLST/monai_cache"
            )

        # datasets
        self.luna = LUNA_Confidence_Flat(luna_args, split_group)
        self.nlst = NLST_Confidence_Flat(nlst_args, split_group)
        self.dataset = self.create_dataset(split_group)
