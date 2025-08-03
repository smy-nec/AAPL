from collections.abc import Sequence
from typing import Literal, TypedDict, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.focal_loss import sigmoid_focal_loss
from tqdm import tqdm

from aapl.config.schema import Config
from aapl.datasets.structures import SparseSupBatch, SparseSupInstance, WeakSupBatch
from aapl.models.aapl_model import AAPLModel, HeadBase


class AAPLHead(HeadBase):
    _requires_init_with_train_data = True
    prototypes: Tensor

    def __init__(
        self,
        embedding_dims: int,
        num_action_classes: int,
        w_video: float,
        w_contrastive: float,
        fg_threshold: float,
        bg_threshold: float,
        topk_denom: float | Sequence[float],
        kernel_size: int = 1,
        dropout_rate: float = 0.0,
        groups: int = 2,
        two_branch_embedder: bool = True,
        positive_only_video_loss: bool = False,
        prototype_momentum: float = 0.1,
        contrastive_temperature: float = 0.1,
        contrastive_version: int = 1,
    ) -> None:
        assert contrastive_version in (1, 2)
        super().__init__()

        if isinstance(topk_denom, Sequence):
            assert len(topk_denom) == 2
            topk_denom = list(topk_denom)
        else:
            topk_denom = [topk_denom, topk_denom]

        self.cas_module = nn.Conv1d(
            embedding_dims * groups,
            num_action_classes * groups,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.attn_module = nn.Conv1d(
            embedding_dims * groups,
            groups,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.embedding_dims = embedding_dims
        self.groups = groups
        self.w_video = w_video
        self.w_contrastive = w_contrastive
        self.fg_threshold = fg_threshold
        self.bg_threshold = bg_threshold
        self.num_action_classes = num_action_classes
        self.topk_denom = topk_denom
        self.two_branch_embedder = two_branch_embedder
        self.positive_only_video_loss = positive_only_video_loss
        self.momentum = prototype_momentum
        self.temperature = contrastive_temperature
        self.contrastive_version = contrastive_version

        embedder_out_dims = embedding_dims * groups * (2 if two_branch_embedder else 1)
        self.register_buffer(
            "prototypes", torch.zeros((num_action_classes, embedder_out_dims))
        )

    @classmethod
    def from_cfg(cls, cfg: Config) -> "AAPLHead":
        assert cfg.model.head.topk_denoms and len(cfg.model.head.topk_denoms) in (1, 2)
        dropout_rate = (
            cfg.model.dropout_rate if cfg.model.dropout_rate is not None else 0
        )
        split = 2 if cfg.model.head.two_branch_embedder else 1
        return cls(
            embedding_dims=cfg.model.dim_embedded_features
            // (cfg.model.embedder.groups * split),
            num_action_classes=len(cfg.dataset.action_class_names),
            w_video=cfg.model.head.video_loss_weight,
            w_contrastive=cfg.model.head.contrastive_loss_weight,
            fg_threshold=cfg.model.head.fg_threshold,
            bg_threshold=cfg.model.head.bg_threshold,
            topk_denom=cfg.model.head.topk_denoms,
            kernel_size=cfg.model.head.kernel_size,
            groups=cfg.model.embedder.groups,
            dropout_rate=dropout_rate,
            two_branch_embedder=cfg.model.head.two_branch_embedder,
            positive_only_video_loss=cfg.model.head.positive_only_video_loss,
            prototype_momentum=cfg.model.head.prototype_momentum,
            contrastive_temperature=cfg.model.head.contrastive_temperature,
            contrastive_version=cfg.model.head.contrastive_version,
        )

    class HeadOut(TypedDict):
        feat_cas: Tensor  # [B, T, g, D]
        feat_attn: Tensor  # [B, T, g, D]
        cas: Tensor  # [B, T, g, C]
        cas_logits: Tensor  # [B, T, g, C]
        attention: Tensor  # [B, T, g, 1]
        attention_logits: Tensor
        fused_cas: Tensor  # [B, T, g, C]

    def forward_head(self, features: Tensor) -> HeadOut:
        if self.two_branch_embedder:
            # Features: [B, T, 2 * g * D] => [B, T, g, D, 2]
            features = features.unflatten(dim=-1, sizes=(self.groups, 2, -1))  # type: ignore[no-untyped-call]
            feat_cas = features.select(dim=-2, index=0).flatten(-2)  # [B, T, g * D]
            feat_attn = features.select(dim=-2, index=1).flatten(-2)  # [B, T, g * D]
        else:
            feat_cas = feat_attn = features

        # [B, T, gD] -> [B, gD, T] -> [B, T, gC] -> [B, T, g, C]
        cas_logits = (
            self.cas_module(self.dropout(feat_cas.permute((0, 2, 1))))
            .permute((0, 2, 1))
            .unflatten(dim=-1, sizes=(self.groups, -1))
        )
        cas = cas_logits.sigmoid()
        attn_logits = (
            self.attn_module(self.dropout(feat_attn.permute((0, 2, 1))))
            .permute((0, 2, 1))
            .unflatten(dim=-1, sizes=(self.groups, -1))
        )
        attn = attn_logits.sigmoid()

        return self.HeadOut(
            feat_cas=feat_cas.unflatten(dim=-1, sizes=(self.groups, -1)),  # type: ignore[no-untyped-call]
            feat_attn=feat_attn.unflatten(dim=-1, sizes=(self.groups, -1)),  # type: ignore[no-untyped-call]
            cas=cas,
            cas_logits=cas_logits,  # [B, T, g, C]
            attention=attn,  # [B, T, g, 1]
            attention_logits=attn_logits,
            fused_cas=cas * attn
            # cluster_assign_attn=cluster_assign_attn,  # [B, T, g, K]
            # ccc_gt_attn=self._ccc_gt_attn,
        )

    def _propagate_point_labels(
        self,
        head_out: HeadOut,
        instance: SparseSupBatch,
        fg_threshold: float,
        bg_threshold: float,
    ) -> tuple[Tensor, Tensor]:
        """This function propagates point labels based on the attention and cas values.
        If a point label representing background is present on an interval with the attention
        values below bg_threshold, then the label is propagated to the entire interval.
        Similarly, if a point label representing a foreground class is present on an interval with
        the CAS values above fg_threshold, then the label is propagated to the entire interval.
        """
        # Averaging over the groups
        attention = head_out["attention"].mean(dim=2)  # [B, T, 1]
        cas = head_out["fused_cas"].mean(dim=2)  # [B, T, C]

        B, T, C = cas.shape
        device = cas.device

        # BG
        bg_mask = attention < bg_threshold  # [B, T, 1], True for bg, False for fg
        # torch.diff yields 1 at the beginning of each interval, -1 at the end, and 0 elsewhere.
        # With abs(), we get 1 at the edges of each interval, and 0 elsewhere.
        interval_bndry_attn = torch.diff(  # [B, T, 1]
            torch.cat((bg_mask[:, :1], bg_mask), dim=1),  # [B, T+1, 1]
            dim=1,
        ).abs()
        interval_idx_attn = (
            torch.cat(  # [B, 1, T + 1] - append 1 at the end of each sequence
                [
                    interval_bndry_attn.permute((0, 2, 1)),  # [B, 1, T]
                    torch.ones((B, 1, 1), device=device),
                ],
                dim=2,
            )
            .flatten()  # [B * (T + 1)]
            .cumsum(dim=0)  # [B * (T + 1)] - to get the interval indices
            .reshape((B, -1, 1))  # [B, T + 1, 1]
            .narrow(dim=1, start=0, length=T)  # [B, T, 1] - remove the appended 1s
        )
        # Propagate the bg label to the intervals with attn scores below bg_threshold
        # and on which bg labels are present
        # Pad the tensors to the same size
        bg_t_indices_padded = torch.nn.utils.rnn.pad_sequence(
            instance["bg_t_indices"], batch_first=True, padding_value=-1
        ).unsqueeze(2)
        labeled_indices = torch.where(  # [B, max_size, 1], -1 for padding
            bg_t_indices_padded >= 0,
            interval_idx_attn.gather(1, bg_t_indices_padded.clamp(min=0)),
            -1,
        )
        # torch.cuda.current_stream().synchronize()
        bg_labels_prop = torch.logical_and(
            bg_mask,
            interval_idx_attn.unsqueeze(2).eq(labeled_indices.unsqueeze(1)).any(2),
        ).float()
        for b in range(B):
            bg_labels_prop[b].index_fill_(0, instance["bg_t_indices"][b], 1)

        # FG (same as BG, but with CAS and classwise)
        fg_mask = torch.logical_and(  # [B, T, C]
            attention > bg_threshold, cas > fg_threshold
        )
        interval_bndry_cas = torch.diff(  # [B, T, C]
            torch.cat(
                (torch.zeros((B, 1, C), dtype=torch.bool, device=device), fg_mask),
                dim=1,
            ),
            dim=1,
        ).abs()
        interval_idx_cas = torch.cumsum(interval_bndry_cas, dim=1)

        fg_labels_prop = torch.zeros_like(cas)  # [B, T, C]
        fg_labels_batch = instance["fg_labels"]
        fg_t_indices_batch = instance["fg_t_indices"]

        for b in range(B):
            fg_labels = fg_labels_batch[b]
            fg_t_indices = fg_t_indices_batch[b]
            labeled_interval_indices = fg_labels * interval_idx_cas[b, fg_t_indices] - (
                1 - fg_labels
            )
            fg_labels_prop[b] = torch.logical_and(
                fg_mask[b],
                torch.any(
                    torch.eq(
                        interval_idx_cas[b].unsqueeze(1),
                        labeled_interval_indices.unsqueeze(0),
                    ),
                    dim=1,
                ),
            )
            fg_labels_prop[b, fg_t_indices] = fg_labels
        # Assertions below must hold if fg_threshold >= 1.0 and bg_threshold <= 0.0
        # assert all(
        #     [
        #         (
        #             fg_labels_prop[b, instance["fg_t_indices"][b]]
        #             == instance["fg_labels"][b]
        #         ).all()
        #         for b in range(instance["batch_size"])
        #     ]
        # )
        # assert all(
        #     fg_labels_prop[b].sum() == instance["fg_labels"][b].sum()
        #     for b in range(instance["batch_size"])
        # )
        # assert all(
        #     (bg_labels_prop[b, instance["bg_t_indices"][b]] == 1).all()
        #     for b in range(instance["batch_size"])
        # )
        # assert all(
        #     bg_labels_prop[b].sum() == len(instance["bg_t_indices"][b])
        #     for b in range(instance["batch_size"])
        # )

        return fg_labels_prop, bg_labels_prop

    def calculate_losses(
        self, head_out: HeadOut, instance: SparseSupBatch  # type: ignore
    ) -> tuple[Tensor, dict[str, Tensor]]:
        fg_labels_prop, bg_labels_prop = self._propagate_point_labels(
            head_out, instance, self.fg_threshold, self.bg_threshold
        )

        loss_fg, loss_bg = self._calculate_base_loss(
            head_out, fg_labels_prop, bg_labels_prop
        )
        loss_video, topk_indices = self._calculate_video_loss(head_out, instance)
        loss = loss_fg + loss_bg + self.w_video * loss_video
        loss_dict = {
            "loss_fg": loss_fg.detach(),
            "loss_bg": loss_bg.detach(),
            "loss_video": loss_video.detach(),
        }

        if self.w_contrastive <= 0:
            return loss, loss_dict

        if self.contrastive_version == 1:
            loss_contrast_fg, loss_contrast_bg = self._calculate_contrastive_loss(
                head_out, fg_labels_prop, bg_labels_prop, temperature=self.temperature
            )
            loss = loss + self.w_contrastive * (loss_contrast_bg + loss_contrast_fg)
            loss_dict["loss_cont_fg"] = loss_contrast_fg.detach()
            loss_dict["loss_cont_bg"] = loss_contrast_bg.detach()
        elif self.contrastive_version == 2:
            loss_contrast = self._calculate_contrastive_loss_v2(
                head_out, fg_labels_prop, bg_labels_prop, temperature=self.temperature
            )
            loss = loss + self.w_contrastive * loss_contrast
            loss_dict["loss_contrast"] = loss_contrast.detach()
        else:
            raise ValueError(f"Invalid contrastive version: {self.contrastive_version}")

        self.update_prototypes(head_out, fg_labels_prop, momentum=self.momentum)

        return loss, loss_dict

    def _calculate_base_loss(
        self, head_out: HeadOut, fg_labels: Tensor, bg_labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the base loss for the SStad head.

        Args:
            head_out (HeadOut): The output of the head module.
            fg_labels (Tensor): The foreground labels. shape: [B, T, C]
            bg_labels (Tensor): The background labels. shape: [B, T, 1]

        Returns:
            tuple[Tensor, Tensor]: The calculated foreground loss and background loss.
        """
        fg_labels_agno = fg_labels.max(dim=-1, keepdim=True)[0]  # [B, T, 1]

        cas_logits, attn_logits = head_out["cas_logits"], head_out["attention_logits"]

        loss_cas_nored = sigmoid_focal_loss(  # [B, T, C]
            cas_logits,
            fg_labels.unsqueeze(2).expand_as(cas_logits),
            gamma=2,
            alpha=-1,
            reduction="none",
        ).sum(dim=2)
        loss_attn_nored = sigmoid_focal_loss(  # [B, T, C]
            attn_logits,
            fg_labels_agno.unsqueeze(2).expand_as(attn_logits),
            gamma=2,
            alpha=-1,
            reduction="none",
        ).sum(dim=2)
        loss_nored = loss_cas_nored + loss_attn_nored

        num_act = fg_labels_agno.sum(dim=1, keepdim=True).clamp(1)
        num_bg = bg_labels.sum(dim=1, keepdim=True).clamp(1)
        loss_fg = (loss_nored * fg_labels_agno / num_act).sum(dim=(1, 2)).mean(dim=0)
        loss_bg = (loss_nored * bg_labels / num_bg).sum(dim=(1, 2)).mean(dim=0)

        return loss_fg, loss_bg

    @overload
    def topk_pooling(
        self,
        pooled_scores: Tensor,
        rank_src: Tensor,
        with_bottom: Literal[False] = False,
    ) -> tuple[Tensor, Tensor]:
        ...

    @overload
    def topk_pooling(
        self, pooled_scores: Tensor, rank_src: Tensor, with_bottom: Literal[True]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ...

    def topk_pooling(
        self, pooled_scores: Tensor, rank_src: Tensor, with_bottom: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        k = max(1, int(rank_src.size(1) / self.topk_denom[0]))
        topk_indices: Tensor = (  # top-k in the temporal dimension, [B, k, 1, C]
            torch.topk(rank_src, k=k, dim=1, sorted=False)[1]
        )
        video_logits = torch.gather(
            pooled_scores,
            1,
            topk_indices.expand((-1, k, self.groups, self.num_action_classes)),
        ).mean(
            dim=1
        )  # [B, g, C]

        if not with_bottom:
            return video_logits, topk_indices

        k = max(1, int(rank_src.size(1) / self.topk_denom[1]))
        btmk_indices: Tensor = (  # top-k in the temporal dimension, [B, k, 1, C]
            torch.topk(rank_src, k=k, dim=1, largest=False, sorted=False)[1]
        )
        btmk_logits = torch.gather(
            pooled_scores,
            1,
            btmk_indices.expand((-1, k, self.groups, self.num_action_classes)),
        ).mean(
            dim=1
        )  # [B, g, C]
        return video_logits, topk_indices, btmk_logits, btmk_indices

    def _calculate_video_loss(
        self, head_out: HeadOut, instance: WeakSupBatch
    ) -> tuple[Tensor, Tensor]:
        fused_cas = head_out["fused_cas"].mean(dim=2, keepdim=True)  # [B, T, 1, C]
        (
            video_logits,
            topk_indices,
            btmk_logits,
            btmk_indices,
        ) = self.topk_pooling(  # [B, g, C], [B, k, 1, C]
            # head_out["cas_logits"], fused_cas
            fused_cas,
            fused_cas,
            with_bottom=True,
        )

        video_labels = (  # [B, g, C]
            instance["video_labels"].unsqueeze(1).expand_as(video_logits)
        )
        if self.positive_only_video_loss:
            loss_video = (
                (
                    video_labels
                    * F.binary_cross_entropy_with_logits(
                        video_logits, video_labels, reduction="none"
                    )
                )
                .sum(dim=(1, 2))
                .mean()
            )
            loss_video = loss_video + (
                F.binary_cross_entropy_with_logits(
                    btmk_logits, torch.zeros_like(btmk_logits), reduction="none"
                )
                .sum(dim=(1, 2))
                .mean()
            )
        else:
            loss_video = (
                F.binary_cross_entropy_with_logits(
                    video_logits, video_labels, reduction="none"
                )
                .sum(dim=(1, 2))
                .mean()
            )

        return loss_video, topk_indices

    def _masked_logsumexp(
        self, x: Tensor, mask: Tensor, dim: int, keepdim: bool = False
    ) -> Tensor:
        x = x.masked_fill(~mask, -torch.inf)
        x_max: Tensor = x.max(dim=dim, keepdim=True)[0]
        return (x - x_max).exp().sum(dim=dim, keepdim=keepdim).log() + (
            x_max if keepdim else x_max.squeeze(-1)
        )

    def _calculate_contrastive_loss(
        self,
        head_out: HeadOut,
        fg_labels_prop: Tensor,  # [B, T, C]
        bg_labels_prop: Tensor,  # [B, T, 1]
        temperature: float,
    ) -> tuple[Tensor, Tensor]:
        prototypes = self.prototypes.detach().unsqueeze(0).unsqueeze(0)  # [1, 1, C, D]
        if self.two_branch_embedder:
            features = torch.cat(
                [
                    head_out["feat_cas"].flatten(-2),
                    head_out["feat_attn"].flatten(-2),
                ],
                dim=-1,
            ).unsqueeze(-2)
        else:
            features = head_out["feat_cas"].flatten(-2).unsqueeze(-2)  # [B, T, 1, D]
        sim_f_p = (  # [B, T, C]
            F.cosine_similarity(features, prototypes, dim=-1) / temperature
        )
        loss_fg = torch.mean(
            torch.sum(
                fg_labels_prop
                * (-sim_f_p + torch.logsumexp(sim_f_p, dim=-1, keepdim=True)),
                dim=(1, 2),
            )
            / fg_labels_prop.sum(dim=(1, 2)).clamp(1)
        )

        bg_lse = self._masked_logsumexp(  # [B, T, C]
            sim_f_p, bg_labels_prop.bool(), dim=1, keepdim=True
        ).expand_as(sim_f_p)
        loss_bg = torch.mean(
            torch.mean(
                bg_labels_prop
                * (
                    -sim_f_p
                    + torch.logsumexp(torch.stack([sim_f_p, bg_lse], dim=-1), dim=-1)
                ),
                dim=2,
            ).sum(dim=1)
            / bg_labels_prop.sum(dim=(1, 2)).clamp(1)
        )
        return loss_fg, loss_bg

    def _calculate_contrastive_loss_v2(
        self,
        head_out: HeadOut,
        fg_labels_prop: Tensor,  # [B, T, C]
        bg_labels_prop: Tensor,  # [B, T, 1]
        temperature: float,
    ) -> Tensor:
        prototypes = self.prototypes.detach().unsqueeze(0)  # [1, C, D]
        if self.two_branch_embedder:
            features = torch.cat(
                [
                    head_out["feat_cas"].flatten(-2),
                    head_out["feat_attn"].flatten(-2),
                ],
                dim=-1,
            )
        else:
            features = head_out["feat_cas"].flatten(-2)  # [B, T, D]
        features = features.flatten(end_dim=1).unsqueeze(-2)  # [B * T, 1, D]
        sim_f_p = (  # [B * T, C]
            F.cosine_similarity(features, prototypes, dim=-1) / temperature
        )
        fg_labels_prop = fg_labels_prop.flatten(end_dim=1)  # [B * T, C]
        bg_labels_prop = bg_labels_prop.flatten(end_dim=1)  # [B * T, 1]
        mask_agnostic = torch.logical_or(  # [B * T, 1]
            fg_labels_prop.any(-1, keepdim=True), bg_labels_prop
        )
        lse = self._masked_logsumexp(sim_f_p, mask_agnostic, dim=0, keepdim=True)
        loss = torch.sum(
            ((-sim_f_p + lse) * fg_labels_prop).sum(dim=0)
            / fg_labels_prop.sum(dim=0).clamp(1)
        )

        return loss

    @torch.no_grad()
    def init_with_train_data(
        self,
        train_data_set: Sequence[SparseSupInstance],
        wstad_model: AAPLModel,
    ) -> None:
        embedding_dims = (
            self.embedding_dims * self.groups * (2 if self.two_branch_embedder else 1)
        )
        prototypes = torch.zeros((self.num_action_classes, embedding_dims)).cuda()
        counts = torch.zeros((self.num_action_classes, 1)).cuda()

        wstad_model.eval()
        wstad_model.cuda()

        for idx in tqdm(range(len(train_data_set))):
            instance = train_data_set[idx]
            embedded = wstad_model.embedder(instance["input"].cuda().unsqueeze(0))[0]

            fg_feats = embedded[instance["fg_t_indices"]]  # [T, (2)gD]
            fg_labels = instance["fg_labels"].cuda()  # [T, C]
            prototypes += fg_labels.t().mm(fg_feats)
            counts += fg_labels.sum(dim=0).unsqueeze(-1)
        prototypes = prototypes / counts.clamp(1)
        self.prototypes.copy_(prototypes)

        wstad_model.cpu()

    @torch.no_grad()
    def update_prototypes(
        self,
        head_out: HeadOut,
        fg_labels: Tensor,  # [B, T, C]
        momentum: float,
    ) -> None:
        if self.two_branch_embedder:
            features = torch.cat(
                [
                    head_out["feat_cas"].flatten(-2),
                    head_out["feat_attn"].flatten(-2),
                ],
                dim=-1,
            )
        else:
            features = head_out["feat_cas"].flatten(-2)  # [B, T, D]
        feats_masked = features.unsqueeze(2) * fg_labels.unsqueeze(3)  # [B, T, C, D]
        num_labels = fg_labels.sum(dim=(0, 1)).unsqueeze(-1)  # [C, 1]
        feats_batch_avg = feats_masked.sum(dim=(0, 1)) / num_labels.clamp(1)  # [C, D]
        self.prototypes = (
            self.prototypes * (1 - momentum * (num_labels > 0).float())
            + feats_batch_avg * momentum
        )
