from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from .activecam_core import CAMERAS
    from .router_core import STATE_FEATURES
except ImportError:  # Direct module execution on the experiment server.
    from activecam_core import CAMERAS
    from router_core import STATE_FEATURES


@dataclass(frozen=True)
class RouterLoss:
    loss: Tensor
    listwise: Tensor
    pairwise: Tensor
    regression: Tensor
    top1_accuracy: Tensor
    mean_regret: Tensor


class BudgetConditionedSetRouter(nn.Module):
    """Score whole camera sets from deployable per-camera memory state."""

    def __init__(
        self,
        *,
        state_dim: int = len(STATE_FEATURES),
        hidden_dim: int = 64,
        camera_embedding_dim: int = 12,
        max_budget: int = len(CAMERAS),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_budget = max_budget
        self.camera_embedding = nn.Embedding(len(CAMERAS), camera_embedding_dim)
        self.budget_embedding = nn.Embedding(max_budget + 1, 8)
        self.camera_encoder = nn.Sequential(
            nn.Linear(state_dim + camera_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim),
            nn.SiLU(),
        )
        # global, selected, unselected, candidate identity, switch fraction, budget ratio
        scorer_dim = hidden_dim * 3 + len(CAMERAS) + 2
        self.set_scorer = nn.Sequential(
            nn.Linear(scorer_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.previous_selected_index = STATE_FEATURES.index("previous_selected")

    def forward(
        self,
        state: Tensor,
        action_masks: Tensor,
        *,
        budget: Tensor,
        feasible_mask: Tensor | None = None,
    ) -> Tensor:
        if state.ndim != 3 or state.shape[1:] != (len(CAMERAS), self.state_dim):
            raise ValueError("state must have shape [batch, 6, state_dim]")
        if action_masks.ndim == 2:
            action_masks = action_masks.unsqueeze(0).expand(state.shape[0], -1, -1)
        if action_masks.ndim != 3 or action_masks.shape[::2] != (state.shape[0], len(CAMERAS)):
            raise ValueError("action_masks must have shape [actions, 6] or [batch, actions, 6]")
        if budget.ndim == 0:
            budget = budget.expand(state.shape[0])
        budget = budget.long().clamp(0, self.max_budget)

        camera_ids = torch.arange(len(CAMERAS), device=state.device)
        camera_ids = camera_ids.unsqueeze(0).expand(state.shape[0], -1)
        encoded = self.camera_encoder(
            torch.cat([state, self.camera_embedding(camera_ids)], dim=-1)
        )
        global_context = self.global_encoder(
            torch.cat([
                encoded.mean(dim=1),
                self.budget_embedding(budget),
            ], dim=-1)
        )
        masks = action_masks.to(dtype=encoded.dtype)
        selected_count = masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        unselected_count = (1.0 - masks).sum(dim=-1, keepdim=True).clamp_min(1.0)
        selected = torch.einsum("bac,bch->bah", masks, encoded) / selected_count
        unselected = torch.einsum("bac,bch->bah", 1.0 - masks, encoded) / unselected_count
        global_repeated = global_context.unsqueeze(1).expand(-1, masks.shape[1], -1)
        previous = state[:, :, self.previous_selected_index].unsqueeze(1)
        switch = torch.abs(masks - previous).mean(dim=-1, keepdim=True)
        budget_ratio = budget.to(encoded.dtype).view(-1, 1, 1) / self.max_budget
        budget_ratio = budget_ratio.expand(-1, masks.shape[1], -1)
        features = torch.cat([
            global_repeated,
            selected,
            unselected,
            masks,
            switch,
            budget_ratio,
        ], dim=-1)
        scores = self.set_scorer(features).squeeze(-1)
        if feasible_mask is not None:
            if feasible_mask.shape != scores.shape:
                raise ValueError("feasible_mask must match candidate scores")
            if not torch.all(feasible_mask.any(dim=1)):
                raise RuntimeError("hard safety shield has no feasible candidate")
            scores = scores.masked_fill(~feasible_mask, -1.0e9)
        return scores


class IndependentCameraRouter(nn.Module):
    """Strong additive Top-K baseline without camera-set interaction terms."""

    def __init__(
        self,
        *,
        state_dim: int = len(STATE_FEATURES),
        hidden_dim: int = 64,
        camera_embedding_dim: int = 12,
        max_budget: int = len(CAMERAS),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_budget = max_budget
        self.camera_embedding = nn.Embedding(len(CAMERAS), camera_embedding_dim)
        self.budget_embedding = nn.Embedding(max_budget + 1, 8)
        self.camera_encoder = nn.Sequential(
            nn.Linear(state_dim + camera_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim),
            nn.SiLU(),
        )
        self.camera_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: Tensor,
        action_masks: Tensor,
        *,
        budget: Tensor,
        feasible_mask: Tensor | None = None,
    ) -> Tensor:
        if state.ndim != 3 or state.shape[1:] != (len(CAMERAS), self.state_dim):
            raise ValueError("state must have shape [batch, 6, state_dim]")
        if action_masks.ndim == 2:
            action_masks = action_masks.unsqueeze(0).expand(state.shape[0], -1, -1)
        if action_masks.ndim != 3 or action_masks.shape[::2] != (state.shape[0], len(CAMERAS)):
            raise ValueError("action_masks must have shape [actions, 6] or [batch, actions, 6]")
        if budget.ndim == 0:
            budget = budget.expand(state.shape[0])
        budget = budget.long().clamp(0, self.max_budget)
        camera_ids = torch.arange(len(CAMERAS), device=state.device)
        camera_ids = camera_ids.unsqueeze(0).expand(state.shape[0], -1)
        encoded = self.camera_encoder(
            torch.cat([state, self.camera_embedding(camera_ids)], dim=-1)
        )
        global_context = self.global_encoder(
            torch.cat([encoded.mean(dim=1), self.budget_embedding(budget)], dim=-1)
        )
        global_context = global_context.unsqueeze(1).expand(-1, len(CAMERAS), -1)
        camera_scores = self.camera_scorer(
            torch.cat([encoded, global_context], dim=-1)
        ).squeeze(-1)
        masks = action_masks.to(dtype=camera_scores.dtype)
        scores = torch.einsum("bac,bc->ba", masks, camera_scores)
        if feasible_mask is not None:
            if feasible_mask.shape != scores.shape:
                raise ValueError("feasible_mask must match candidate scores")
            if not torch.all(feasible_mask.any(dim=1)):
                raise RuntimeError("hard safety shield has no feasible candidate")
            scores = scores.masked_fill(~feasible_mask, -1.0e9)
        return scores


def router_ranking_loss(
    scores: Tensor,
    teacher_q: Tensor,
    feasible_mask: Tensor,
    *,
    temperature: float = 0.10,
    pairwise_weight: float = 0.20,
    regression_weight: float = 0.0,
    margin: float = 0.20,
) -> RouterLoss:
    if temperature <= 0 or margin < 0 or pairwise_weight < 0 or regression_weight < 0:
        raise ValueError("loss hyperparameters are invalid")
    if scores.shape != teacher_q.shape or scores.shape != feasible_mask.shape:
        raise ValueError("scores, q values, and mask must have identical shapes")
    if not torch.all(feasible_mask.any(dim=1)):
        raise RuntimeError("training item has no feasible candidate")
    masked_q = teacher_q.masked_fill(~feasible_mask, -1.0e9)
    # Per-state centering avoids scale drift from varying object counts.
    finite_q = torch.where(feasible_mask, masked_q, torch.zeros_like(masked_q))
    mean_q = finite_q.sum(dim=1, keepdim=True) / feasible_mask.sum(dim=1, keepdim=True)
    target_prob = F.softmax((masked_q - mean_q) / temperature, dim=1)
    log_prob = F.log_softmax(scores.masked_fill(~feasible_mask, -1.0e9), dim=1)
    listwise = -(target_prob * log_prob).sum(dim=1).mean()

    best = masked_q.argmax(dim=1)
    best_score = scores.gather(1, best.unsqueeze(1)).squeeze(1)
    negative_mask = feasible_mask.clone()
    negative_mask.scatter_(1, best.unsqueeze(1), False)
    hard_negative = scores.masked_fill(~negative_mask, -1.0e9).max(dim=1).values
    has_negative = negative_mask.any(dim=1)
    pair_terms = F.relu(margin - best_score + hard_negative)
    pairwise = pair_terms[has_negative].mean() if has_negative.any() else scores.new_zeros(())
    regression = F.smooth_l1_loss(scores[feasible_mask], teacher_q[feasible_mask])
    loss = listwise + pairwise_weight * pairwise + regression_weight * regression

    predicted = scores.masked_fill(~feasible_mask, -1.0e9).argmax(dim=1)
    predicted_q = masked_q.gather(1, predicted.unsqueeze(1)).squeeze(1)
    best_q = masked_q.gather(1, best.unsqueeze(1)).squeeze(1)
    return RouterLoss(
        loss=loss,
        listwise=listwise,
        pairwise=pairwise,
        regression=regression,
        top1_accuracy=(predicted == best).float().mean(),
        mean_regret=(best_q - predicted_q).mean(),
    )
