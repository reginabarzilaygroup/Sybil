import torch
import torch.nn as nn
import torch.nn.functional as F
from sybil.utils.sliding_window_inference import sliding_window_inference_custom
from sybil.utils.augmentations import PatchAugmentations, min_max_normalize_batch
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class nnUNet(nn.Module):
    def __init__(self, args):
        super(nnUNet, self).__init__()
        nn_args = {
            "n_stages": 6,
            "features_per_stage": [32, 64, 128, 256, 320, 320],
            "conv_op": torch.nn.modules.conv.Conv3d,
            "kernel_sizes": [
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            "strides": [
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": torch.nn.modules.instancenorm.InstanceNorm3d,
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": torch.nn.LeakyReLU,
            "nonlin_kwargs": {"inplace": True},
        }

        self.model = ResidualEncoderUNet(
            input_channels=args.num_chan, num_classes=args.num_classes, **nn_args
        )
        if args.module_snapshot is not None:
            weights = torch.load(args.module_snapshot, weights_only=False)
            if "hyper_parameters" in weights:
                weights = {
                    k[len("model.") :]: v for k, v in weights["state_dict"].items()
                }
                weights = {
                    k[len("model.") :]: v for k, v in weights.items() if "model." in k
                }
                weights = {"network_weights": weights}

            for key in [
                "encoder.stem.convs.0.conv.weight",
                "encoder.stem.convs.0.all_modules.0.weight",
                "decoder.encoder.stem.convs.0.conv.weight",
                "decoder.encoder.stem.convs.0.all_modules.0.weight",
            ]:
                weights["network_weights"][key] = weights["network_weights"][key][
                    :, : args.num_chan
                ]

            if args.num_classes != 2:
                for key in [
                    "decoder.seg_layers.0.weight",
                    "decoder.seg_layers.0.bias",
                    "decoder.seg_layers.1.weight",
                    "decoder.seg_layers.1.bias",
                    "decoder.seg_layers.2.weight",
                    "decoder.seg_layers.2.bias",
                    "decoder.seg_layers.3.weight",
                    "decoder.seg_layers.3.bias",
                    "decoder.seg_layers.4.weight",
                    "decoder.seg_layers.4.bias",
                ]:
                    original_weight = weights["network_weights"][key]
                    if original_weight.shape[0] < args.num_classes:
                        # Repeat weights to achieve num_classes dimension
                        repeat_factor = (
                            args.num_classes + original_weight.shape[0] - 1
                        ) // original_weight.shape[0]
                        repeated_weight = original_weight.repeat(
                            repeat_factor, *[1] * (len(original_weight.shape) - 1)
                        )
                        weights["network_weights"][key] = repeated_weight[
                            : args.num_classes
                        ]
                    else:
                        # Truncate if we have more weights than needed
                        weights["network_weights"][key] = original_weight[
                            : args.num_classes
                        ]
            self.model.load_state_dict(weights["network_weights"])

        if args.use_object_classifier:
            self.use_object_classifier = True
            self.classifier = nn.ModuleList()
            for chan in [32, 64, 128, 256, 320, 320]:
                self.classifier.append(nn.Linear(chan, 2))

        self.roi_size = (args.anatomix_crop_size[-1],) + tuple(
            args.anatomix_crop_size[:-1]
        )
        self.inference_augmentations = PatchAugmentations(args, split="predict")
        self.args = args

    def forward(self, x, batch=None):
        if self.args.predict:
            outputs = self.predict(x, batch)
            predicted_scores = F.softmax(outputs, 1)
            outputs = {
                "pred_mask_logit": outputs,
                "pred_mask": predicted_scores,  # prob score
                "hidden": predicted_scores[:, 1],
                "pred_masks_pos": 1 * (predicted_scores[:, -1] > 0.5),  # binary
            }
        else:
            skips = self.model.encoder(x)
            outputs = self.model.decoder(skips)

            predicted_scores = F.softmax(outputs, 1)
            outputs = {
                "logit": outputs,
                "pred_mask": predicted_scores,  # prob score
                "pred_masks_pos": 1 * (predicted_scores[:, -1] > 0.5),  # binary
                "hidden": predicted_scores,
                # "losses": losses,
            }

        return outputs

    @torch.no_grad()
    def predict(self, x, batch=None, sw_batch_size=82):
        outputs = sliding_window_inference_custom(
            inputs=x,
            predictor=self.model,
            roi_size=self.roi_size,
            overlap=0.5,
            sw_batch_size=sw_batch_size,
            progress=False,
            augmentations=min_max_normalize_batch,  # self.inference_augmentations,
        )

        return outputs


class nnUNetConfidence(nn.Module):
    def __init__(self, args):
        super(nnUNetConfidence, self).__init__()
        nn_args = {
            "n_stages": 6,
            "features_per_stage": [32, 64, 128, 256, 320, 320],
            "conv_op": torch.nn.modules.conv.Conv3d,
            "kernel_sizes": [
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            "strides": [
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": torch.nn.modules.instancenorm.InstanceNorm3d,
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": torch.nn.LeakyReLU,
            "nonlin_kwargs": {"inplace": True},
        }

        self.model = ResidualEncoderUNet(
            input_channels=args.num_chan, num_classes=2, **nn_args
        )
        if args.module_snapshot is not None:
            weights = torch.load(args.module_snapshot, weights_only=False)
            for key in [
                "encoder.stem.convs.0.conv.weight",
                "encoder.stem.convs.0.all_modules.0.weight",
                "decoder.encoder.stem.convs.0.conv.weight",
                "decoder.encoder.stem.convs.0.all_modules.0.weight",
            ]:
                weights["network_weights"][key] = weights["network_weights"][key][
                    :, : args.num_chan
                ]
            self.model.load_state_dict(weights["network_weights"])

        self.model = self.model.encoder

        self.classifier = nn.ModuleList()
        for chan in [32, 64, 128, 256, 320, 320]:
            self.classifier.append(nn.Linear(chan, 2))
        self.args = args

    def forward(self, x, batch=None):
        if (
            self.args.dataset
            in ["nlst_sparse_confidence", "nlst_sparse_confidence_sybiltest"]
        ) and (self.args.batch_size == 1):
            x = x[0]
        skips = self.model(x)

        # Use the classifier to compute detection score
        detection_score = 0
        for i, hidden in enumerate(skips):
            detection_score = detection_score + self.classifier[i](
                torch.amax(hidden, dim=(2, 3, 4))
            )
        if (
            self.args.dataset
            in ["nlst_sparse_confidence", "nlst_sparse_confidence_sybiltest"]
        ) and (self.args.batch_size == 1):
            detection_score = detection_score.unsqueeze(0)
        outputs = {"logit": detection_score.as_tensor()}

        return outputs
