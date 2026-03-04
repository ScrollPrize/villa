import torch
import torch.nn as nn


class GroupDROComputer(nn.Module):
    def __init__(
        self,
        n_groups,
        group_counts,
        *,
        alpha=None,
        gamma=0.1,
        adj=None,
        min_var_weight=0.0,
        step_size=0.01,
        normalize_loss=False,
        btl=False,
    ):
        super().__init__()
        self.n_groups = int(n_groups)
        self.gamma = float(gamma)
        self.alpha = alpha
        self.min_var_weight = float(min_var_weight)
        self.step_size = float(step_size)
        self.normalize_loss = bool(normalize_loss)
        self.btl = bool(btl)

        group_counts = torch.as_tensor(group_counts, dtype=torch.float)
        if group_counts.numel() != self.n_groups:
            raise ValueError(
                f"group_counts must have length {self.n_groups}, got {int(group_counts.numel())}"
            )
        self.register_buffer("group_counts", group_counts)
        self.register_buffer("group_frac", group_counts / group_counts.sum().clamp_min(1))

        if adj is None:
            adj = torch.zeros(self.n_groups, dtype=torch.float)
        else:
            adj = torch.as_tensor(adj, dtype=torch.float)
        if adj.numel() != self.n_groups:
            raise ValueError(f"adj must have length {self.n_groups}, got {int(adj.numel())}")
        self.register_buffer("adj", adj)

        self.register_buffer("adv_probs", torch.ones(self.n_groups, dtype=torch.float) / self.n_groups)
        self.register_buffer("exp_avg_loss", torch.zeros(self.n_groups, dtype=torch.float))
        self.register_buffer("exp_avg_initialized", torch.zeros(self.n_groups, dtype=torch.bool))

    def compute_group_avg(self, losses, group_idx):
        group_idx = group_idx.long()
        group_map = (group_idx == torch.arange(self.n_groups, device=group_idx.device).unsqueeze(1).long()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * self.exp_avg_initialized.float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss.mul_(prev_weights).add_(group_loss * curr_weights)
        self.exp_avg_initialized |= group_count > 0

    def compute_robust_loss(self, group_loss):
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss = adjusted_loss + self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / adjusted_loss.sum()

        with torch.no_grad():
            self.adv_probs.mul_(torch.exp(self.step_size * adjusted_loss.detach()))
            self.adv_probs.div_(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss):
        adjusted_loss = self.exp_avg_loss
        if torch.all(self.adj > 0):
            adjusted_loss = adjusted_loss + self.adj / torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        if self.alpha is None:
            raise ValueError("alpha must be specified when btl=True")

        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= float(self.alpha)
        weights = mask.float() * sorted_frac / float(self.alpha)
        last_idx = int(mask.sum().item())
        if last_idx >= self.n_groups:
            last_idx = self.n_groups - 1
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (1 - self.min_var_weight)

        robust_loss = sorted_loss @ weights

        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def loss(self, per_sample_losses, group_idx):
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        self.update_exp_avg_loss(group_loss, group_count)

        if self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss)
        else:
            actual_loss, weights = self.compute_robust_loss(group_loss)

        return actual_loss, group_loss, group_count, weights

