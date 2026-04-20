from dataclasses import dataclass
from itertools import chain
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aemvc_fusion import AEMVCFusion
from network import GAT, GCN, GraphLearn, VLTransformer


@dataclass
class EvalOutput:
    logits: torch.Tensor
    total_loss: torch.Tensor
    mmgl_loss: torch.Tensor
    aemvc_loss: torch.Tensor
    repr: torch.Tensor


class EvalHelper(nn.Module):
    # FUSION MODIFICATION: joint AEMVC-MMGL evaluator with backward-compatible no-fusion path.
    def __init__(self, args, modal_dims: Dict[str, int], num_classes: int = 2):
        super().__init__()
        self.args = args
        self.modal_dims = dict(modal_dims)
        self.use_fusion = args.impute_mode in {"fusion", "full_aemvc", "simple_aemvc"}
        self.use_joint = args.impute_mode == "fusion"
        self.aemvc_alpha = float(args.aemvc_alpha)
        self.use_kernel_concat = not getattr(args, "disable_kernel_concat", False)

        # FUSION MODIFICATION: AEMVC fusion module initialization.
        self.aemvc_module = AEMVCFusion(
            modal_dims=modal_dims,
            latent_dim=args.aemvc_latent_dim,
            kernel_type=args.aemvc_kernel_type,
            lambda1=args.aemvc_lambda1,
            lambda2=args.aemvc_lambda2,
            lambda3=args.aemvc_lambda3,
            use_kcca=args.aemvc_use_kcca,
        )

        modal_input_dims = {k: v + (1 if self.use_kernel_concat else 0) for k, v in modal_dims.items()}
        self.transformer = VLTransformer(modal_input_dims=modal_input_dims, hidden_dim=args.hidden_dim, use_kernel_concat=self.use_kernel_concat)
        self.graph_learn = GraphLearn(input_dim=args.hidden_dim, use_prior_fusion=args.use_graph_prior_fusion, init_beta=args.graph_prior_beta)
        self.gnn = GAT(args.hidden_dim, args.hidden_dim, num_classes) if args.gnn_type == "gat" else GCN(args.hidden_dim, args.hidden_dim, num_classes)

        self.main_optimizer = torch.optim.Adam(
            # FUSION MODIFICATION: chain parameter iterables to avoid accidental duplication patterns.
            chain(self.transformer.parameters(), self.graph_learn.parameters(), self.gnn.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.aemvc_optimizer = torch.optim.Adam(self.aemvc_module.parameters(), lr=args.aemvc_lr, weight_decay=args.weight_decay)

    def _make_prior_adj_from_kernels(self, kernels: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not kernels:
            return None
        mats = list(kernels.values())
        prior = torch.mean(torch.stack(mats, dim=0), dim=0)
        prior = torch.clamp(prior, min=0.0)
        return prior

    def forward(self, modal_data_dict: Dict[str, torch.Tensor], y: torch.Tensor, missing_mask_dict: Optional[Dict] = None) -> EvalOutput:
        kernel_repr = None
        aemvc_loss = torch.tensor(0.0, device=y.device)
        mmgl_inputs = modal_data_dict

        if self.use_fusion:
            imputed_data, kernel_repr, aemvc_loss = self.aemvc_module(modal_data_dict, missing_mask_dict or {})
            mmgl_inputs = imputed_data

        patient_repr = self.transformer(mmgl_inputs, kernel_repr_dict=kernel_repr if self.use_kernel_concat else None)
        prior_adj = self._make_prior_adj_from_kernels(kernel_repr or {}) if self.args.use_graph_prior_fusion else None
        adj = self.graph_learn(patient_repr, prior_adj=prior_adj)
        logits = self.gnn(patient_repr, adj)
        cls_loss = F.cross_entropy(logits, y.long())

        # FUSION MODIFICATION: MMGL regularization terms preserved in configurable form.
        smooth = torch.trace(patient_repr.t() @ (torch.diag(torch.sum(adj, dim=1)) - adj) @ patient_repr) / max(1, patient_repr.shape[0])
        degree = torch.mean((torch.sum(adj, dim=1) - 1.0) ** 2)
        sparsity = torch.mean(torch.abs(adj))
        aux = torch.tensor(0.0, device=logits.device)
        mmgl_loss = cls_loss + self.args.theta_smooth * smooth + self.args.theta_degree * degree + self.args.theta_sparsity * sparsity + self.args.eta * aux

        total_loss = mmgl_loss + (self.aemvc_alpha * aemvc_loss if self.use_joint else 0.0 * aemvc_loss)
        return EvalOutput(logits=logits, total_loss=total_loss, mmgl_loss=mmgl_loss, aemvc_loss=aemvc_loss, repr=patient_repr)

    def run_epoch(
        self,
        modal_data_dict: Dict[str, torch.Tensor],
        y: torch.Tensor,
        missing_mask_dict: Optional[Dict] = None,
        train: bool = True,
        pretrain_aemvc_only: bool = False,
    ) -> EvalOutput:
        if train:
            self.main_optimizer.zero_grad()
            self.aemvc_optimizer.zero_grad()
        out = self.forward(modal_data_dict=modal_data_dict, y=y, missing_mask_dict=missing_mask_dict)
        loss = out.aemvc_loss if pretrain_aemvc_only else out.total_loss
        if train:
            loss.backward()
            if pretrain_aemvc_only:
                self.aemvc_optimizer.step()
            else:
                if self.use_joint:
                    self.aemvc_optimizer.step()
                self.main_optimizer.step()
        return out
