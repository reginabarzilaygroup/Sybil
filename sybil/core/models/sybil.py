from sandstone.models.factory import load_model, RegisterModel, get_model_by_name
from sandstone.models.pools.factory import get_pool
import math
import torch
import torch.nn as nn
import pdb
import numpy as np
import copy
from models.cumulative_probability_layer import Cumulative_Probability_Layer
import time
import json
import torchvision
import torch.nn.functional as F
from sandstone.utils.nlst_risk_factors import NLSTRiskFactorVectorizer
import warnings
import pdb

@RegisterModel("r3d")
class R3D18(nn.Module):
    def __init__(self, args):
        super(R3D18, self).__init__()

        self.args = args
        encoder = torchvision.models.video.r3d_18(pretrained=args.pretrained_on_imagenet)
        num_features = encoder.fc.in_features
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])

        pool_name = args.pool_name
        self.pool = get_pool(pool_name)(args, args.hidden_dim)
        if not self.pool.replaces_fc():
            # Cannot not placed on self.all_blocks since requires intermediate op
            self.relu = nn.ReLU(inplace=False)
            self.dropout = nn.Dropout(p=args.dropout)
            self.fc = nn.Linear(args.hidden_dim, args.num_classes)
        
        if args.survival_analysis_setup and not args.eval_risk_on_survival:
            self.prob_of_failure_layer = Cumulative_Probability_Layer(args.hidden_dim, args, max_followup=args.max_followup)
        
        if args.use_annotations:
           assert args.region_annotations_filepath, 'ANNOTATIONS METADATA FILE NOT SPECIFIED'
           self.annotations_metadata = json.load(open(args.region_annotations_filepath, 'r'))
        
        
    def forward(self, x, batch=None):
        output = {}
        x = self.image_encoder(x)
        pool_output = self.aggregate_and_classify(x, batch=batch)
        output['activ'] = x
        output.update(pool_output)

        return output

    def aggregate_and_classify(self, x, batch):
        
        pool_output =  self.pool(x, batch)

        for pool_name in self.pool.pool_names:
            if self.args.multipool_pools is not None:
                hidden_key, logit_key = '{}_hidden'.format(pool_name), '{}_logit'.format(pool_name)
            else:
                hidden_key, logit_key = 'hidden', 'logit'
            
            if not self.pool.replaces_fc():
                pool_output[hidden_key] = self.relu(pool_output[hidden_key])
                pool_output[hidden_key] = self.dropout(pool_output[hidden_key])
                pool_output[logit_key]  = self.fc(pool_output[hidden_key])

            if self.args.survival_analysis_setup and not (hasattr(self.args, 'eval_risk_on_survival') and  self.args.eval_risk_on_survival):
                pool_output[logit_key] = self.prob_of_failure_layer(pool_output[hidden_key])
            
        return pool_output


@RegisterModel("risk_factors_predictor")
class RiskFactorPredictor(R3D18):
    def __init__(self, args):
        super(RiskFactorPredictor, self).__init__(args)

        self.length_risk_factor_vector = NLSTRiskFactorVectorizer(args).vector_length
        for key in args.risk_factor_keys:
            num_key_features = args.risk_factor_key_to_num_class[key]
            key_fc = nn.Linear(args.hidden_dim, num_key_features)
            self.add_module('{}_fc'.format(key), key_fc)

    def forward(self, x, batch):
        output = {}
        x = self.image_encoder(x)
        output = self.pool(x, batch)
        
        if self.args.multipool_pools is not None:
            hidden_key, logit_key = '{}_hidden'.format(self.pool.pool_names[0]), '{}_logit'.format(self.pool.pool_names[0])
        else:
            hidden_key, logit_key = 'hidden', 'logit'
        
        hidden = output[hidden_key]

        risk_factors = batch['risk_factors']
        for indx, key in enumerate(self.args.risk_factor_keys):
            output['{}_logit'.format(key)] = self._modules['{}_fc'.format(key)](hidden)

        return output

    def get_loss_functions(self, args):
        return ['risk_factor_loss']
