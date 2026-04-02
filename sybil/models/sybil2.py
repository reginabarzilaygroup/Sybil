import pickle
import torch
import torch.nn as nn
from collections import OrderedDict
from torch_scatter import scatter_max
from sybil.models.pillar.multi_stage import BaseMultiStage
from sybil.models.set_transformer import SetTransformer
from sybil.models.segformer3d import SegFormer3DModel
import loguru as logger


class MultiStage(BaseMultiStage):
    def __init__(self, args):
        kwargs = args.model["kwargs"]
        super().__init__(args, **kwargs)

    def forward(self, x, batch=None, split="train"):
        # Pass through backbone and get features
        backbone_outputs = OrderedDict(self.backbone_model.forward(x, batch=batch))
        features = backbone_outputs[
            "activ"
        ]  # shape is: [batch_size, backbone_hidden_dim, D, H, W]
        pooled_features = backbone_outputs.get("pooled", None)

        # Process heads
        head_outputs, pooled_outputs = self._process_heads(
            features, pooled_features, split=split
        )

        # Merge outputs
        merged_outputs = OrderedDict()
        merged_outputs.update(backbone_outputs)
        merged_outputs.update(pooled_outputs)
        merged_outputs.update(head_outputs)

        backbone_outputs.clear()
        pooled_outputs.clear()
        head_outputs.clear()

        return merged_outputs

    def no_weight_decay(self):
        return []

    def stages(self):
        return self.backbone_model.visual.model_config["model"]["stages"]


class DiffNet1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = 512  # args.diffnet_dim
        # project global features to diffnet model hidden dim - 6 (for logit)
        self.global_feature_proj = nn.Linear(1152, self.hidden_dim - 6)
        # project nodule features to diffnet model hidden dim - 3 (for malignancy score, volume, and segmentation confidence)
        self.nodule_feature_proj = nn.Linear(512, self.hidden_dim - 3)
        # nodule entity embedding
        self.entity_embedding = nn.Embedding(100, self.hidden_dim)
        # nodule timepoint embedding, continuous
        self.timepoint_embedding = nn.Linear(1, self.hidden_dim)
        # transformer layer
        self.transformer_layer = SetTransformer(
            embed_dim=512,
            num_heads=4,
            num_layers=args.diffnet_num_layers,
            ffn_dim=None,
            dropout=0.1,
            use_isab=False,
            num_inducing_points=5,
            num_outputs=1,
            output_dim=args.max_followup,
        )
        self.nodule_classifer = nn.Linear(self.hidden_dim, 1)
        self.global_risk = nn.Linear(self.hidden_dim, args.max_followup)

    def forward(self, inputs):
        # extract global embeddings
        gfeatures = self.global_feature_proj(inputs["gfeatures"])
        gfeatures = torch.concat([gfeatures, inputs["glogit"]], dim=-1)
        # extract nodule embeddings
        nfeatures = self.nodule_feature_proj(inputs["nfeatures"])
        nfeatures = torch.concat(
            [nfeatures, inputs["nlogits"], inputs["nvolumes"].unsqueeze(-1)], dim=-1
        )
        # add entity and timepoint embeddings
        nitems = gfeatures.shape[0]
        global_entity_emb = self.entity_embedding(
            torch.zeros(nitems, dtype=torch.long, device=nfeatures.device)
        )
        global_time = torch.hstack(
            [
                inputs["nodule_tps"].max(1).values.unsqueeze(1),
                inputs["nodule_tps"].min(1).values.unsqueeze(1),
            ]
        )
        global_time = global_time.float().view(-1, 1)
        global_time_emb = self.timepoint_embedding(
            torch.tensor(global_time.float(), device=nfeatures.device).unsqueeze(1)
        )
        global_time_emb = global_time_emb.view(nitems, -1, self.hidden_dim)
        nodule_entity_emb = self.entity_embedding((inputs["nodule_ids"] + 1).int())
        nodule_tps = inputs["nodule_tps"].float().view(-1, 1)
        nodule_time_emb = self.timepoint_embedding(nodule_tps)
        nodule_time_emb = nodule_time_emb.view(nitems, -1, self.hidden_dim)

        # combine nodule features with embeddings
        gfeatures = gfeatures + global_entity_emb.unsqueeze(1) + global_time_emb
        nfeatures = nfeatures + nodule_entity_emb + nodule_time_emb

        # mask out dummy nodules
        dummy_nodules = inputs["old_nodule_ids"] == -1
        dummy_global = torch.zeros(
            (nitems, 1), dtype=torch.bool, device=nfeatures.device
        )
        dummy_global = torch.hstack(
            [dummy_global, ~inputs["has_prior"].bool().unsqueeze(-1)]
        )
        dummy_mask = torch.concat([dummy_global, dummy_nodules], dim=1)

        # combine global and nodule features
        features = torch.cat(
            [gfeatures, nfeatures], dim=1
        )  # shape: [num_nodules + 1, hidden_dim]

        features = self.transformer_layer(features, padding_mask=dummy_mask)
        nodule_risks = self.nodule_classifer(features["hidden"][:, 2:])
        global_risk = self.global_risk(features["hidden"][:, :2])
        return {
            "logit": features["output"],
            "nodule_risks": nodule_risks,
            "global_risk": global_risk,
        }


class SegFormerClassifier(nn.Module):
    def __init__(self, args):
        super(SegFormerClassifier, self).__init__()
        self.model = SegFormer3DModel(
            in_channels=args.num_chan,
            num_classes=args.num_classes,
            decoder_dropout=args.dropout,
        )
        del self.model.decoder
        self.classifier = nn.Linear(512, args.num_classes)

    def forward(self, x, batch=None):
        encoder_hidden_states = []
        output_attentions = None

        # Process through encoder stages
        for stage_idx in range(4):
            # Track input spatial dimensions
            if stage_idx == 0:
                spatial_shape = x.shape[2:]  # (D, H, W)
            # Patch embedding
            x = self.model.encoders[stage_idx](x)
            spatial_shape = self.model.encoders[stage_idx].get_output_shape(
                spatial_shape
            )
            B, N, C = x.shape
            # Transformer blocks
            for block in self.model.transformer_blocks[stage_idx]:
                block_outputs, spatial_shape = block(
                    x, spatial_shape, output_attentions
                )
                x = block_outputs[0]
            # Layer norm
            x = self.model.encoder_norms[stage_idx](x)
            # Reshape and store hidden state using calculated dimensions
            d, h, w = spatial_shape
            x_reshaped = x.reshape(B, d, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
            encoder_hidden_states.append(x)
            # Prepare input for next stage if not last stage
            if stage_idx < 3:
                x = x_reshaped

        hidden = [torch.mean(hs, 1) for hs in encoder_hidden_states]
        x = torch.cat(hidden, dim=1)
        pred = self.classifier(x)  # B, num_classes
        output = {"logit": pred, "features": x}
        return output


class Sybil17(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.precomputed_pillar_hiddens = args.precomputed_pillar_hiddens
        # self._load_pillar_model(args)
        self._load_nodule_model(args, args.nodule_classifier_ckpt)
        self._load_diff_net(args, args.diffnet_ckpt)

    def _load_pillar_model(self, args):
        self.pillar_model = MultiStage(args)
        checkpoint = torch.load(
            args.pillar_ckpt,
            map_location="cpu",
            weights_only=False,
        )
        self.pillar_model.load_state_dict(checkpoint["model"])
        self.pillar_model.eval()

    def _load_nodule_model(self, args, ckpt_path=None):
        if ckpt_path is None:
            self.nodule_model = SegFormerClassifier(args)
        else:
            nodule_model_ckpt = torch.load(
                ckpt_path,
                weights_only=False,
                map_location="cpu",
            )
            args = nodule_model_ckpt["hyper_parameters"]["args"]
            self.nodule_model = SegFormerClassifier(args)
            self.nodule_model.load_state_dict(
                {
                    k[len("model.") :]: v
                    for k, v in nodule_model_ckpt["state_dict"].items()
                }
            )

    def _load_diff_net(self, args, ckpt_path=None):
        if ckpt_path is None:
            self.diff_net = DiffNet1(args)
        else:
            diff_net_ckpt = torch.load(ckpt_path, weights_only=False)
            args = diff_net_ckpt["hyper_parameters"]["args"]
            self.diff_net = DiffNet1(args)
            self.diff_net.load_state_dict(diff_net_ckpt["state_dict"])

    def forward(self, x, batch=None, split="train"):
        output = self.forward_multi_batch(x, batch, split=split)
        return output

    def forward_multi_batch(self, x, batch=None, split="train"):
        batch["nodule_ids_tracked"] = batch["nodule_ids_tracked"].long()
        assert self.precomputed_pillar_hiddens, (
            "Multi-batch only supported with precomputed pillar hiddens"
        )
        pillar_output = {
            "pillar_features": x,
            "pillar_risk": batch["logit"],
        }
        # get nodule features
        nodule_x = batch["nodule_x"].view(-1, 1, *batch["nodule_x"].shape[2:])
        nodule_output = self.nodule_forward(nodule_x)
        nodule_output = {
            k: v.view(batch["nodule_x"].shape[0], -1, *v.shape[1:])
            for k, v in nodule_output.items()
        }

        # diff net
        diff_output = self.diffnet_forward(pillar_output, nodule_output, batch=batch)

        # reshape
        diff_output["nodule_risks"] = diff_output["nodule_risks"].view(-1, 1)

        output = {
            "logit": diff_output["logit"],
            "nodule_risks": diff_output["nodule_risks"],
            "global_risk": diff_output["global_risk"],
            "batch_ids": batch["nodule_batch_id"][0],
        }
        return output

    def nodule_forward(self, x):
        return self.nodule_model(x)

    def diffnet_forward(self, global_features, nodule_features, batch):
        diffnet_inputs = {
            "gfeatures": global_features["pillar_features"],
            "glogit": global_features["pillar_risk"],
            "nfeatures": nodule_features["features"],
            "nlogits": nodule_features["logit"],
            "nsegmentation_confidence": batch["nodule_confidence"],
            "nodule_ids": batch["nodule_ids_tracked"],
            "nodule_tps": batch["nodule_tp_id"],
            "nvolumes": batch["nodule_volumes"],
            "old_nodule_ids": batch["old_nodule_ids"],
            "has_prior": batch["has_prior"],
            "nodule_batch_id": batch["nodule_batch_id"],
        }
        return self.diff_net(diffnet_inputs)

    @torch.no_grad()
    def pillar_forward(self, x, batch):
        backbone_outputs = OrderedDict(
            self.pillar_model.backbone_model.forward(x, batch=batch)
        )
        features = backbone_outputs[
            "activ"
        ]  # shape is: [batch_size, backbone_hidden_dim, D, H, W]
        # get pooled features and risk
        pooled_outputs = OrderedDict(self.pillar_model.pool(features))
        dropout_outputs = self.pillar_model.dropout(pooled_outputs["hidden"])
        logit = self.pillar_model.head_models["survival"](dropout_outputs)
        output = {
            "pillar_risk": logit,
            "pillar_features": pooled_outputs["hidden"],
        }
        return output
