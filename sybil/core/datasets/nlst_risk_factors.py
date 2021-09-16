import numpy as np
import pdb
import copy
import torch


MISSING_VALUE = -1
HASNT_HAPPENED_VALUE = -5

RACE_CODE_TO_NAME = {
        1: 'White',
        2: 'African American',
        3: 'Asian',
        4: 'American Indian or Alaskan Native',
        5: 'Native Hawaiian or Other Pacific Islander',
        6: 'More than one race'
        #7: 'Unknown',
        #95: 'Unknown',
        #96: 'Unknown',
        #98: 'Unknown',
        #99: 'Unknown'
}

TREAT_MISSING_AS_NEGATIVE = False
NEGATIVE_99 = -99

class NLSTRiskFactorVectorizer():
    def __init__(self, args):
        # cutoffs: exclude population min and max
        self.risk_factor_transformers = {
                        'gender': self.get_gender_transform,
                        'age': self.get_age_risk_factor_transformer('age', [55, 60, 65, 70, 75]),
                        'race': self.transform_race,
                        'weight': self.get_exam_one_hot_risk_factor_transformer('weight', [155, 180, 210]),
                        'height': self.get_exam_one_hot_risk_factor_transformer('height', [65, 68, 71]),
                        'binary_family_history': self.transform_binary_family_history,
                        'copd': self.get_binary_transformer('diagcopd'),
                        'is_smoker': self.get_binary_transformer('cigsmok'),
                        'smoking_duration': self.transform_smoking_tails('smokeyr', [30, 50]),
                        'smoking_intensity': self.transform_smoking_tails('smokeday', [20, 40]),
                        'years_since_quit_smoking': self.get_years_since_quit_smoking_transformer('years_since_quit_smoking', [0, 10 ]),
                        }

        self.risk_factor_keys = args.risk_factor_keys
        self.feature_names = []
        self.risk_factor_key_to_num_class = {}
        for k in self.risk_factor_keys:
            if k not in self.risk_factor_transformers.keys():
                raise Exception("Risk factor key '{}' not supported.".format(k))
            names = self.risk_factor_transformers[k](None, None, just_return_feature_names=True)
            self.risk_factor_key_to_num_class[k] = len(names)
            self.feature_names.extend(names)
        args.risk_factor_key_to_num_class = self.risk_factor_key_to_num_class

    @property
    def vector_length(self):
        return len(self.feature_names)

    def get_feature_names(self):
        return copy.deepcopy(self.feature_names)

    def one_hot_vectorizor(self, value, cutoffs):
        one_hot_vector = torch.zeros(len(cutoffs) + 1)
        if value == MISSING_VALUE:
            return one_hot_vector
        for i, cutoff in enumerate(cutoffs):
            if value <= cutoff:
                one_hot_vector[i] = 1
                return one_hot_vector
        one_hot_vector[-1] = 1
        return one_hot_vector

    def one_hot_feature_names(self, risk_factor_name, cutoffs):
        feature_names = [""] * (len(cutoffs) + 1)
        feature_names[0] = "{}_lt_{}".format(risk_factor_name, cutoffs[0])
        feature_names[-1] = "{}_gt_{}".format(risk_factor_name, cutoffs[-1])
        for i in range(1, len(cutoffs)):
            feature_names[i] = "{}_{}_{}".format(risk_factor_name, cutoffs[i - 1], cutoffs[i])
        return feature_names


    def get_age_risk_factor_transformer(self, risk_factor_key, cutoffs):
        def transform_age_risk_factor(patient_factors, screen_timepoint, just_return_feature_names=False):
            if just_return_feature_names:
                return self.one_hot_feature_names(risk_factor_key, cutoffs)
            age_at_randomization = patient_factors['age'][0]
            days_since_randomization = patient_factors['scr_days{}'.format(screen_timepoint)][0]
            exam_age = age_at_randomization + days_since_randomization//365
            return self.one_hot_vectorizor(exam_age, cutoffs)
        return transform_age_risk_factor

    def transform_binary_family_history(self, patient_factors, screen_timepoint, just_return_feature_names=False):
        if just_return_feature_names:
            return (['no_family_history', 'has_family_history'])

        binary_risk_factor = torch.zeros(2)
        risk_factor = int(any( [ patient_factors[key][0] == 1  for key in patient_factors if key.startswith('fam') ]))
        binary_risk_factor[risk_factor] = 1
        return binary_risk_factor

        return binary_family_history


    def transform_smoking_tails(self,  risk_factor_key, cutoffs):
        def transform_smoking_risk_factors(patient_factors, screen_timepoint, just_return_feature_names=False):
            l,u = cutoffs[0], cutoffs[-1]
            if just_return_feature_names:
                return (['<={}'.format(l), '{}<='.format(u)])

            binary_risk_factor = torch.zeros(2)
            if int(patient_factors[risk_factor_key][0]) <= l:
                binary_risk_factor[0] = 1
            elif int(patient_factors[risk_factor_key][0]) >= u:
                binary_risk_factor[1] = 1
            return binary_risk_factor

        return transform_smoking_risk_factors
    
    def get_exam_one_hot_risk_factor_transformer(self, risk_factor_key, cutoffs):
        def transform_exam_one_hot_risk_factor(patient_factors, screen_timepoint, just_return_feature_names=False):
            if just_return_feature_names:
                return self.one_hot_feature_names(risk_factor_key, cutoffs)
            risk_factor = int(patient_factors[risk_factor_key][0])
            return self.one_hot_vectorizor(risk_factor, cutoffs)

        return transform_exam_one_hot_risk_factor

    def get_years_since_quit_smoking_transformer(self, risk_factor_key, cutoffs):
        def transform_exam_one_hot_risk_factor(patient_factors, screen_timepoint, just_return_feature_names=False):
            l,u = cutoffs[0], cutoffs[-1]
            if just_return_feature_names:
                return (['<={}'.format(l), '{}<='.format(u)])
                #return self.one_hot_feature_names(risk_factor_key, cutoffs)

            age_at_randomization = patient_factors['age'][0]
            days_since_randomization = patient_factors['scr_days{}'.format(screen_timepoint)][0]
            current_age = age_at_randomization + days_since_randomization//365

            age_quit_smoking = patient_factors['age_quit'][0]
            is_smoker = patient_factors['cigsmok'][0]

            years_since_quit_smoking = 0  if is_smoker else current_age - age_quit_smoking
            binary_risk_factor = torch.zeros(2)
            
            if is_smoker:
                return binary_risk_factor
            if years_since_quit_smoking <= cutoffs[0]:
                binary_risk_factor[0] =1
            elif years_since_quit_smoking >= cutoffs[1]:
                binary_risk_factor[1]=1
            return binary_risk_factor
            #return self.one_hot_vectorizor(years_since_quit_smoking, cutoffs)

        return transform_exam_one_hot_risk_factor

    def get_binary_transformer(self, risk_factor_key):
        def transform_binary(patient_factors, screen_timepoint, just_return_feature_names=False):
            if just_return_feature_names:
                return ['no_{}'.format(risk_factor_key), 'has_{}'.format(risk_factor_key)]
            binary_risk_factor = torch.zeros(2)
            risk_factor = int(patient_factors[risk_factor_key][0])
            if risk_factor != MISSING_VALUE:
                binary_risk_factor[risk_factor] = 1
            return binary_risk_factor

        return transform_binary

    def get_gender_transform(self, patient_factors, screen_timepoint, just_return_feature_names=False):
        if just_return_feature_names:
            return ['male', 'female']
        binary_risk_factor = torch.zeros(2)
        risk_factor = int(patient_factors['gender'][0])
        if risk_factor in [1,2]:
            binary_risk_factor[risk_factor-1] = 1
        return binary_risk_factor


    def transform_race(self, patient_factors, screen_timepoint, just_return_feature_names=False):
        values = range(1, 7)
        race_vector = torch.zeros(len(values))
        if just_return_feature_names:
            return [RACE_CODE_TO_NAME[i] for i in values]
        race = int(patient_factors['race'][0])
        if race in RACE_CODE_TO_NAME:
            race_vector[race - 1] = 1
        return race_vector

    def transform(self, patient_factors, screen_timepoint):
        risk_factor_vecs = [self.risk_factor_transformers[key](patient_factors, screen_timepoint) for key in self.risk_factor_keys]
        return risk_factor_vecs

    def get_risk_factors_for_sample(self, patient_metadata, screen_timepoint):
        risk_factor_vector = self.transform(patient_metadata, screen_timepoint)
        return risk_factor_vector
