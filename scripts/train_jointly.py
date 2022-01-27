from collections import OrderedDict
from argparse import Namespace
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pytorch_lightning as pl
import torch

from sybil.utils.helpers import get_dataset
import sybil.utils.losses as losses
import sybil.utils.metrics as metrics
import sybil.utils.loading as loaders
import sybil.models as models

from sybil.parsing import parse_args
from scripts.train import (
    SybilLightning,
    prefix_dict
    )

class SybilLightningAdapt(SybilLightning):
    """
    Lightning Module
    Methods:
        .log/.log_dict: log inputs to logger
    Notes:
        *_epoch_end method returns None
        self can log additional data structures to logger
        with self.logger.experiment.log_*
        (*= 'text', 'image', 'audio', 'confusion_matrix', 'histogram')
    """

    def __init__(self, args):
        super(SybilLightningAdapt, self).__init__(args)
        if isinstance(args, dict):
            args = Namespace(**args)
        args.hidden_dim = self.model.hidden_dim
        self.args = args
        self.discriminator = models.adversary.AlignmentMLP(args)
        self.save_prefix = "default"
        self.save_hyperparameters(args)
        self._list_of_metrics = [
            metrics.get_classification_metrics,
            metrics.get_survival_metrics,
            metrics.get_risk_metrics,
            metrics.get_alignment_metrics
        ]

    def step(self, batch, batch_idx, optimizer_idx, log_key_prefix=""):
        logging_dict, predictions_dict = OrderedDict(), OrderedDict()

        if optimizer_idx == 0 or optimizer_idx is None:
            self.reverse_discrim_loss = True
            model_output = self(batch['x'])

            if 'exam' in batch:
                predictions_dict['exam'] = batch['exam']
            if 'y' in batch:
                predictions_dict['golds'] = batch['y']

            if self.args.save_attention_scores:
                attentions = {k: v for k, v in model_output.items() if any([attn_key in k for attn_key in ['attention', 'coord'] ]) }
                predictions_dict.update(attentions)
            
            loss_fns = self.get_loss_functions(self.args)
            loss = 0
            for loss_fn in loss_fns:
                local_loss, local_log_dict, local_predictions_dict = loss_fn(
                    model_output, batch, self, self.args
                )
                loss += local_loss
                logging_dict.update(local_log_dict)
                predictions_dict.update(local_predictions_dict)

        # train discriminator
        elif optimizer_idx == 1:
            self.reverse_discrim_loss = False
            with torch.no_grad():
                model_output = self.model(batch['x'])

            loss, local_log_dict, local_predictions_dict = losses.discriminator_loss(model_output, batch, self, self.args)
            logging_dict.update(local_log_dict)
            predictions_dict.update(local_predictions_dict)

        else:
            print("Got invalid optimizer_idx! optimizer_idx =", optimizer_idx)

        logging_dict = prefix_dict(logging_dict, log_key_prefix)
        predictions_dict = prefix_dict(predictions_dict, log_key_prefix)

        return loss, logging_dict, predictions_dict, model_output

    def configure_optimizers(self):
        """
        Helper function to fetch optimizer based on args.
        """
        params1 = [param for param in self.model.parameters() if param.requires_grad]
        params2 = [param for param in self.discriminator.parameters() if param.requires_grad]
        if self.args.optimizer == "adam":
            optimizer1 = torch.optim.Adam(
                params1, lr=self.args.lr, weight_decay=self.args.weight_decay
            )
            optimizer2 = torch.optim.Adam(
                params2, lr=self.args.adv_lr, weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == "adagrad":
            optimizer1 = torch.optim.Adagrad(
                params1, lr=self.args.lr, weight_decay=self.args.weight_decay
            )
            optimizer2 = torch.optim.Adagrad(
                params2, lr=self.args.adv_lr, weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == "sgd":
            optimizer1 = torch.optim.SGD(
                params1,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum,
            )
            optimizer2 = torch.optim.SGD(
                params2,
                lr=self.args.adv_lr,
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum,
            )
        else:
            raise Exception("Optimizer {} not supported!".format(self.args.optimizer))

        scheduler1 = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer1,
                patience=self.args.patience,
                factor=self.args.lr_decay,
                mode="min" if "loss" in self.args.tuning_metric else "max",
            ),
            "monitor": "val_{}".format(self.args.tuning_metric),
            "interval": "epoch",
            "frequency": 1,
        }

        scheduler2 = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer2,
                patience=self.args.patience,
                factor=self.args.lr_decay,
                mode="min" if "loss" in self.args.tuning_metric else "max",
            ),
            "monitor": "val_{}".format(self.args.tuning_metric),
            "interval": "epoch",
            "frequency": self.args.num_adv_steps,
        }
        
        return [optimizer1, optimizer2], [scheduler1, scheduler2]

    def get_loss_functions(self, args):
        loss_fns = [losses.get_survival_loss, losses.discriminator_loss]
        if args.use_annotations:
            loss_fns.append(losses.get_annotation_loss)

        return loss_fns

def train(args):
    if not args.turn_off_checkpointing:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.save_dir,
            save_top_k=1,
            verbose=True,
            monitor="val_{}".format(args.tuning_metric)
            if args.tuning_metric is not None
            else None,
            save_last=True,
            mode="min"
            if args.tuning_metric is not None and "loss" in args.tuning_metric
            else "max",
        )
        args.callbacks = [checkpoint_callback]
    trainer = pl.Trainer.from_argparse_args(args)
    # Remove callbacks from args for safe pickling later
    args.callbacks = None
    args.num_nodes = trainer.num_nodes
    args.num_processes = trainer.num_processes
    args.world_size = args.num_nodes * args.num_processes
    args.global_rank = trainer.global_rank
    args.local_rank = trainer.local_rank

    train_dataset = loaders.get_train_dataset_loader(
        args, get_dataset(args.dataset, "train", args)
    )
    dev_dataset = loaders.get_eval_dataset_loader(
        args, get_dataset(args.dataset, "dev", args), False
    )

    args.censoring_distribution = metrics.get_censoring_dist(train_dataset.dataset)
    module = SybilLightningAdapt(args)

    # print args
    for key, value in sorted(vars(args).items()):
        print("{} -- {}".format(key.upper(), value))

    trainer.fit(module, train_dataset, dev_dataset)
    args.model_path = trainer.checkpoint_callback.best_model_path

    print("Saving args to {}".format(args.results_path))
    pickle.dump(vars(args), open(args.results_path, "wb"))

def test(args):
    
    trainer = pl.Trainer.from_argparse_args(args)
    # Remove callbacks from args for safe pickling later
    args.callbacks = None
    args.num_nodes = trainer.num_nodes
    args.num_processes = trainer.num_processes
    args.world_size = args.num_nodes * args.num_processes
    args.global_rank = trainer.global_rank
    args.local_rank = trainer.local_rank

    train_dataset = loaders.get_train_dataset_loader(
        args, get_dataset(args.dataset, "train", args)
    )
    test_dataset = loaders.get_eval_dataset_loader(
        args, get_dataset(args.dataset, "test", args), False
    )

    args.censoring_distribution = metrics.get_censoring_dist(train_dataset.dataset)
    module = SybilLightningAdapt(args)
    module = module.load_from_checkpoint(checkpoint_path= args.snapshot)
    module.args = args

    # print args
    for key, value in sorted(vars(args).items()):
        print("{} -- {}".format(key.upper(), value))

    trainer.test(module, test_dataset)

    print("Saving args to {}".format(args.results_path))
    pickle.dump(vars(args), open(args.results_path, "wb"))

if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train(args)
    elif args.test:
        test(args)
