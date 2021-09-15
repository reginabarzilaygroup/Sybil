from sandstone.learn.losses.factory import RegisterLoss
import torch
import torch.nn.functional as F
import torch.nn as nn
from sandstone.utils.generic import get_base_model_obj
from collections import OrderedDict
from sandstone.utils.generic import log
import pdb

EPSILON = 1e-6

@RegisterLoss("cross_entropy")
def get_cross_entropy_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    total_loss = 0

    pool_names = [''] if not hasattr(model.model , 'pool') else model.model.pool.pool_names
    for pool_name in pool_names:
        if args.multipool_pools is not None:
            key = '{}_'.format(pool_name)
        else:
            key = ''

        logit = model_output['{}logit'.format(key)]
        if args.lightning_name == 'private' and 'y_onehot' in batch:
            logprobs = F.log_softmax(logit, dim=-1)
            batchloss = - torch.sum( batch['y_onehot'] * logprobs, dim=1)
            loss = batchloss.mean()
        else:
            loss = F.cross_entropy(logit, batch['y'].long())

        total_loss += loss
        logging_dict['{}cross_entropy_loss'.format(key)] = loss.detach()
        predictions['{}probs'.format(key)] = F.softmax(logit, dim=-1).detach()
        predictions['{}golds'.format(key)] = batch['y']
    return total_loss * args.primary_loss_lambda, logging_dict, predictions

@RegisterLoss("mse")
def get_mse_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    pred = model_output['logit']
    loss = F.mse_loss(pred, batch['y'].view_as(pred))
    logging_dict['mse_loss'] = loss.detach()
    predictions['pred'] = pred.detach()
    return loss * args.primary_loss_lambda, logging_dict, predictions


@RegisterLoss("survival")
def get_survival_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    total_loss = 0

    pool_names = [''] if not hasattr(model.model , 'pool') else model.model.pool.pool_names
    for pool_name in pool_names:
        if args.multipool_pools is not None:
            key = '{}_'.format(pool_name)
        else:
            key = ''

        assert args.survival_analysis_setup
        logit = model_output['{}logit'.format(key)]
        y_seq, y_mask = batch['y_seq'], batch['y_mask']
        loss = F.binary_cross_entropy_with_logits(logit, y_seq.float(), weight=y_mask.float(), size_average=False)/ torch.sum(y_mask.float())
        total_loss += loss
        logging_dict['{}survival_loss'.format(key)] = loss.detach()
        predictions['{}probs'.format(key)] = F.sigmoid(logit).detach()
        predictions['{}golds'.format(key)] = batch['y']
        predictions['censors'] = batch['time_at_event']
    return total_loss * args.primary_loss_lambda, logging_dict, predictions

@RegisterLoss("pred_rf_loss")
def get_pred_rf_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    assert args.pred_risk_factors
    hidden, risk_factors = model_output['hidden'], batch['risk_factors']
    loss = model.model.pool.get_pred_rf_loss(hidden, batch['risk_factors'])
    logging_dict['pred_rf_loss'] = loss.detach()
    return loss * args.pred_risk_factors_lambda, logging_dict, predictions

@RegisterLoss("linearizing_loss")
def get_linearizing_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    loss = model.model.pool.get_linearizing_loss(model_output['activ'], batch)
    logging_dict['linearizing_loss'] = loss.detach()
    return loss * args.linearizing_lambda , logging_dict, predictions

@RegisterLoss("drug_activ")
def get_drug_activ_loss(model_output, batch, model, args):
    assert args.pred_drug_activ
    logging_dict, predictions = OrderedDict(), OrderedDict()
    logit = model_output['drug_activ_logit']
    y, y_mask = batch['drug_activ'], batch['drug_activ_known']
    loss = F.binary_cross_entropy_with_logits(logit, y.float(), weight=y_mask.float(), size_average=False)/ torch.sum(y_mask.float())
    predictions['activ_probs'] = torch.sigmoid(logit).detach()
    logging_dict['drug_activ_loss'] = loss.detach()
    return loss * args.drug_active_lambda, logging_dict, predictions

