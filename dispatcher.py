import argparse
import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import multiprocessing
import pickle
import csv
import json
import sys
from os.path import dirname, realpath
import random
from collections import OrderedDict
import copy
sys.path.append(dirname(dirname(realpath(__file__))))

from sandstone.utils import parsing
from sandstone.utils.generic import md5

EXPERIMENT_CRASH_MSG = "ALERT! job:[{}] has crashed! Check logfile at:[{}]"
CONFIG_NOT_FOUND_MSG = "ALERT! {} config {} file does not exist!"
RESULTS_PATH_APPEAR_ERR = 'results_path should not appear in config. It will be determined automatically per job'
SUCESSFUL_SEARCH_STR = "SUCCESS! Grid search results dumped to {}"


LOG_KEYS = ['results_path', 'model_path', 'log_path']

parser = argparse.ArgumentParser(description='Sandstone Grid Search Dispatcher. For use information, see `doc/README.md`')
parser.add_argument("--experiment_config_path", required=True, type=str, help="Path of experiment config")
parser.add_argument('--log_dir', type=str, default="logs", help="path to store logs and detailed job level result files")
parser.add_argument('--result_path', type=str, default="results/grid_search.csv", help="path to store grid_search table. This is preferably on shared storage")
parser.add_argument('--sort_key', type=str, default="val_loss", help="How to sort csv")
parser.add_argument('--rerun_experiments', action='store_true', default=False, help='whether to rerun experiments with the same result file location')
parser.add_argument('--shuffle_experiment_order', action='store_true', default=False, help='whether to shuffle order of experiments')
parser.add_argument('--dry_run', action='store_true', default=False, help='whether to not actually run the jobs')


def launch_experiment(gpu, worker_args, flag_string):
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    Alert of something goes wrong.
    :gpu: gpu to run this machine on.
    :flag_string: flags to use for this model run. Will be fed into
    scripts/main.py
    '''
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    log_name = md5(flag_string)
    log_stem = os.path.join(args.log_dir, log_name)
    log_path = '{}.txt'.format(log_stem)
    results_path = "{}.results".format(log_stem)

    experiment_string = "CUDA_VISIBLE_DEVICES={} python -u scripts/main.py {} --results_path {}".format(
        gpu, flag_string, results_path)

    if 'port' in worker_args:
        experiment_string += ' --master_port {}'.format(worker_args['port'])

    if 'host' in worker_args:
        experiment_string += ' --master_host {}'.format(worker_args['host'])


    # forward logs to logfile
    if "--resume" in flag_string and not args.rerun_experiments:
        pipe_str = ">>"
    else:
        pipe_str = ">"

    shell_cmd = "{} {} {} 2>&1".format(experiment_string, pipe_str, log_path)
    print("Launched exp: {}".format(shell_cmd))

    if not os.path.exists(results_path) or args.rerun_experiments:
        if not args.dry_run:
            subprocess.call(shell_cmd, shell=True)

    if not os.path.exists(results_path):
        # running this process failed, alert me
        job_fail_msg = EXPERIMENT_CRASH_MSG.format(experiment_string, log_path)
        print(job_fail_msg)

    return results_path, log_path


def worker(gpu, worker_args, job_queue, done_queue):
    '''
    Worker thread for each gpu. Consumes all jobs and pushes results to done_queue.
    :gpu - gpu this worker can access.
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(gpu, worker_args, params))

def get_summary_dict_from_run(result_path, log_path, experiment_axies):
    summary_dict = OrderedDict()
    try:
        result_dict = pickle.load(open(result_path, 'rb'))
    except Exception as e:
        print("Experiment failed! Logs are located at: {}".format(log_path))
        return summary_dict
    
    result_dict['log_path'] = log_path
    
    for k in experiment_axies:
        summary_dict[k] = result_dict[k]

    splits = []
    for key in ['eval_train', 'dev','test']:
        if result_dict[key]:
            splits.append(key)
    for split in splits:
        split_path = '{}.{}_.metrics'.format(result_path, split)
        stats = pickle.load(open(split_path, 'rb'))
        for k,v in stats.items():
            if isinstance(v, float) or isinstance(v, int):
                summary_dict["{}_{}".format(split, k)] = v

    for k in LOG_KEYS:
        summary_dict[k] = result_dict[k]
    return summary_dict

def collate_all_summmaries(summary_dict, summary_dict_list, experiment_axies, args):
    if len(summary_dict_list) == 0:
        summary_columns = list(summary_dict.keys())
        summary_dict_list = [summary_dict]
        return summary_dict_list, summary_columns

    ## Assume that summary dict list already has all columns merged
    prior_result = summary_dict_list[-1]
    prior_columns = list(prior_result.keys())
    current_columns = list(summary_dict.keys())
    summary_columns = copy.deepcopy(experiment_axies)
    for split in ['train','dev','test']:
        for col_list in [prior_columns, current_columns]:
            for key in col_list:
                if split in key and not key in summary_columns:
                    summary_columns.append(key)
    summary_columns.extend(LOG_KEYS)
    summary_dict_list.append(summary_dict)

    for summary in summary_dict_list:
        for key in summary_columns:
            if not key in summary:
                summary[key] = 'NA'
    if args.sort_key in summary_dict_list[-1]:
        summary_dict_list = sorted(summary_dict_list, key=lambda k: k[args.sort_key])
    else:
        print("Warning: Sort key {} not seen in result files".format(args.sort_key))
    return summary_dict_list, summary_columns

def update_summary_with_results(result_path, log_path, experiment_axies,  summary_dict_list, args):
    assert result_path is not None
    summary_dict = get_summary_dict_from_run(result_path, log_path, experiment_axies)
    summary_dict_list, summary_columns = collate_all_summmaries(summary_dict, summary_dict_list, experiment_axies, args)

    result_dir = os.path.dirname(args.result_path)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # Write summary to csv
    with open(args.result_path, 'w') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=summary_columns)
        writer.writeheader()
        for experiment in summary_dict_list:
            writer.writerow(experiment)
    return summary_dict_list

if __name__ == "__main__":

    args = parser.parse_args()
    if not os.path.exists(args.experiment_config_path):
        print(CONFIG_NOT_FOUND_MSG.format("experiment", args.experiment_config_path))
        sys.exit(1)
    experiment_config = json.load(open(args.experiment_config_path, 'r'))

    if 'results_path' in experiment_config['search_space']:
        print(RESULTS_PATH_APPEAR_ERR)
        sys.exit(1)

    job_list, experiment_axies = parsing.parse_dispatcher_config(experiment_config)
    if args.shuffle_experiment_order:
        random.shuffle(job_list)
    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for job in job_list:
        job_queue.put(job)
    print("Launching Dispatcher with {} jobs!".format(len(job_list)))
    print()
    for worker_indx, gpu in enumerate(experiment_config['available_gpus']):
        print("Start gpu worker {}".format(gpu))
        worker_args = {}
        if 'ports' in experiment_config:
            worker_args['port'] = experiment_config['ports'][worker_indx]
        if 'hosts' in experiment_config:
            worker_args['host'] = experiment_config['hosts'][worker_indx]
        multiprocessing.Process(target=worker, args=(gpu, worker_args, job_queue, done_queue)).start()
    print()

    summary = []

    for i in range(len(job_list)):
        result_path, log_path = done_queue.get()
        #summary = update_summary_with_results(result_path, log_path, experiment_axies, summary, args)
        try:
            result_dict = pickle.load(open(result_path, 'rb'))
            dump_result_string = SUCESSFUL_SEARCH_STR.format(args.result_path)
        except Exception as e:
            print("Experiment failed! Logs are located at: {}".format(log_path))
        print("({}/{}) \t {}".format(i+1, len(job_list), dump_result_string))
