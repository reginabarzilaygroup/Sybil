from copy import copy
from sybil.datasets.mgh import MGH_Dataset
from sybil.datasets.nlst import NLST_Survival_Dataset
from collections import Counter


class MGH_NLST_Combined_Dataset(NLST_Survival_Dataset):
    """
    MGH and NLST Combined.

    Notes:
        - args.img_dir is only used by NLST, and should be set to the top-level directory of the NLST images.
        - args.metadata_path must be a folder that contains both the MGH and NLST metadata (under two seperate folders).

    Options:
        - args.balance_by_dataset can be set in order to sample an equal amount of samples from each dataset during training.
        - args.discard_from_combined_dataset can be set to either 'mgh' or 'nlst', in order to produce a dataset containing 
            only samples with the other. This allows you to evaluate on one dataset while using the alignment_ct lightning module.
    """

    def __init__(self, args, split_group):
        super(MGH_NLST_Combined_Dataset, self).__init__(args, split_group)

    def create_dataset(self, split_group, img_dir):
        """
        Gets the dataset from the paths and labels in the json.
        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
            img_dir(str): The path to the directory containing the images.
        Returns:
            The dataset as a dictionary with img paths, label, 
            and additional information regarding exam or participant
        """
        dataset = []

        mgh_args = copy(self.args)
        mgh_args.dataset_file_path = '/Mounts/rbg-storage1/datasets/MGH_Lung_Fintelmann/mgh_metadata.json'
        mgh_dataset = MGH_Dataset(mgh_args, split_group)

        for exam_dict in mgh_dataset.dataset:
            exam_dict['origin_dataset'] = 0
            dataset.append(exam_dict)

        nlst_args = copy(self.args)
        nlst_args.dataset_file_path = '/Mounts/rbg-storage1/datasets/NLST/full_nlst_google.json'
        nlst_dataset = NLST_Survival_Dataset(nlst_args, split_group)

        for exam_dict in nlst_dataset.dataset:
            exam_dict['origin_dataset'] = 1
            dataset.append(exam_dict)

        return dataset
    
    @property
    def METADATA_FILENAME(self):
        return 'MGH_Lung_Fintelmann/mgh_metadata.json'

    @property
    def task(self):
        return 'CANCER'
    
    @staticmethod
    def set_args(args):
        args.num_classes = args.max_followup
        args.multi_image = True

    def get_summary_statement(self, dataset, split_group):
        summary = "Constructed Combined MGH+NLST CT {} Survival {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
        class_balance = Counter([d['y'] for d in dataset])
        exams = set([d['exam'] for d in dataset])
        patients = set([d['pid'] for d in dataset])
        statement = summary.format(self.task, split_group, len(dataset), len(exams), len(patients), class_balance)
        statement += "\n" + "Censor Times: {}".format( Counter([d['time_at_event'] for d in dataset]))
        annotation_msg = self.annotation_summary_msg(dataset) if self.args.use_annotations else ''
        statement += annotation_msg
        return statement
