import math
from tqdm import tqdm
import pickle


class RiskModel(object):
    def __init__(self, args):
        self.args = args

    def forward(self, batch):
        x_transformed = {
            key: func(batch[key]) for key, func in self.input_transformers.items()
        }
        x_scaled = self.scale_inputs(x_transformed)
        risk = self.model(x_scaled)
        return risk

    def test(self, data):
        results = []
        for sample in tqdm(data.dataset):
            sample["golds"] = sample["y"]
            sample["probs"] = self.forward(sample)

        if self.args.save_predictions:
            self.save_predictions(data.dataset)

    def save_predictions(self, data):
        predictions_dict = [
            {k: v for k, v in d.items() if k in self.save_keys} for d in data
        ]
        predictions_filename = "{}.{}.predictions".format(
            self.args.results_path, self.save_prefix
        )
        pickle.dump(predictions_dict, open(predictions_filename, "wb"))

    @property
    def input_coef(self):
        pass

    @property
    def input_transformers(self):
        pass

class PLCOm2012(RiskModel):
    def __init__(self, args):
        super(PLCOm2012, self).__init__(args)

    def model(self, x):
        return 1 / (1 + math.exp(-x))

    def scale_inputs(self, x):
        running_sum = -4.532506
        for key, beta in self.input_coef.items():
            if key == "race":
                running_sum += beta[x["race"]]
            else:
                running_sum += x[key] * beta
        return running_sum

    @property
    def input_coef(self):
        coefs = {
            "age": 0.0778868,
            "race": {
                "white": 0,
                "black": 0.3944778,
                "hispanic": -0.7434744,
                "asian": -0.466585,
                "native_hawaiian_pacific": 0,
                "american_indian_alaskan": 1.027152,
            },
            "education": -0.0812744,
            "bmi": -0.0274194,
            "cancer_hx": 0.4589971,
            "family_lc_hx": 0.587185,
            "copd": 0.3553063,
            "is_smoker": 0.2597431,
            "smoking_intensity": -1.822606,
            "smoking_duration": 0.0317321,
            "years_since_quit_smoking": -0.0308572,
        }
        return coefs

    @property
    def input_transformers(self):
        funcs = {
            "age": lambda x: x - 62,
            "race": lambda x: x,
            "education": lambda x: x - 4,
            "bmi": lambda x: x - 27,
            "cancer_hx": lambda x: x,
            "family_lc_hx": lambda x: x,
            "copd": lambda x: x,
            "is_smoker": lambda x: x,
            "smoking_intensity": lambda x: 10 / x - 0.4021541613,
            "smoking_duration": lambda x: x - 27,
            "years_since_quit_smoking": lambda x: x - 10,
        }
        return funcs

    @property
    def save_keys(self):
        return [
            "pid",
            "age",
            "race",
            "education",
            "bmi",
            "cancer_hx",
            "family_lc_hx",
            "copd",
            "is_smoker",
            "smoking_intensity",
            "smoking_duration",
            "years_since_quit_smoking",
            "exam",
            "golds",
            "probs",
            "time_at_event",
            "y_seq",
            "y_mask",
            "screen_timepoint",
        ]
