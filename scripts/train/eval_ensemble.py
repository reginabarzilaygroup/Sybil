from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
import argparse
from argparse import Namespace
import pickle
from sandstone.learn.metrics.factory import get_metric
import numpy as np
import os

LOGGER_KEYS = ['censors', 'golds'] 

def make_logging_dict(results):
    logging_dict = {}
    logging_dict = {k: np.array( results[0].get('test_{}'.format(k), 0) ) for k in LOGGER_KEYS}
    logging_dict['probs'] = []
    prob_key = [k.split('_probs')[0] for k in results[0].keys() if 'probs' in k][0]
    
    logging_dict['probs'] = np.mean([np.array(r['{}_probs'.format(prob_key)])  for r in results ], axis = 0)
    
    return logging_dict

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', type = str, default = '/Mounts/rbg-storage1/logs/lung_ct/')
parser.add_argument('--result_file_names', type = str, nargs = '+', default =  ["7a07ee56c93e2abd100a47542e394bed","1e94034923b44462203cfcf29ae29061","9ff7a8ba3b1f9eb7216f35222b3b8524","18490328d1790f7b6b8c86d97f25103c"])
parser.add_argument('--test_suffix', type = str, default = 'test')
parser.add_argument('--metric_name', nargs= '*', type = str, default = 'survival')


if __name__ == '__main__':
    args = parser.parse_args()
    result_args = [Namespace(**pickle.load(open(os.path.join(args.parent_dir, '{}.results'.format(f)), 'rb'))) for f in args.result_file_names]
    test_full_paths = [os.path.join(args.parent_dir, '{}.results.{}_.predictions'.format(f, args.test_suffix)) for f in args.result_file_names]
    test_results = [pickle.load(open(f, 'rb')) for f in  test_full_paths]

    logging_dict = make_logging_dict(test_results)
    performance_dict = {}
    metrics = [get_metric(m) for m in args.metric_name]
    for m in metrics:
        performance_dict.update( m(logging_dict, result_args[0]) )

    print(performance_dict)

