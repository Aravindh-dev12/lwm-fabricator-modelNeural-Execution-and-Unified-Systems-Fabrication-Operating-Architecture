"""Latent World Model (LeWM) — JEPA-based OS state prediction + CEM planning per paper Section 4."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import List, Tuple, Optional


class StateEncoder(nn.Module):
    """Encodes 64D OS state vector into 192D latent representation.
    z_t = GELU(W_e * s_t + b_e), W_e in R^{192x64}"""
    def __init__(self, state_dim: int = 64, latent_dim: int = 192):
        super().__init__()
        self.W_e = nn.Linear(state_dim, latent_dim)
        # Xavier-like initialization: Wij ~ N(0, sqrt(2/(64+192)))
        nn.init.normal_(self.W_e.weight, mean=0.0, std=np.sqrt(2.0 / (state_dim + latent_dim)))
        nn.init.zeros_(self.W_e.bias)

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.W_e(s_t))


class ActionEmbedder(nn.Module):
    """Encodes 32D action vector into 192D latent space.
    a_tilde = SiLU(W_a * a_t + b_a), W_a in R^{192x32}"""
    def __init__(self, action_dim: int = 32, latent_dim: int = 192):
        super().__init__()
        self.W_a = nn.Linear(action_dim, latent_dim)
        nn.init.normal_(self.W_a.weight, mean=0.0, std=np.sqrt(2.0 / (action_dim + latent_dim)))
        nn.init.zeros_(self.W_a.bias)

    def forward(self, a_t: torch.Tensor) -> torch.Tensor:
        return F.silu(self.W_a(a_t))


class TransformerPredictor(nn.Module):
    """Transformer-based autoregressive predictor per paper Section 4.2.3 + config:
    depth=6, heads=16, MLP dim=2048. Takes history of H_ctx=3 latent states + action embedding.
    z_{t+1} = 0.7 * z_t + 0.3 * TransformerPredictor(z_{t-2:t}, a_tilde_t)"""
    def __init__(self, latent_dim: int = 192, depth: int = 6, heads: int = 16, mlp_dim: int = 2048, history_size: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.history_size = history_size

        # Token embedding: each token is a latent state or action embedding (192D)
        self.pos_emb = nn.Parameter(torch.randn(1, history_size + 1, latent_dim) * 0.02)

        # Transformer encoder: depth=6 layers, heads=16, MLP dim=2048
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Output projection: produce predicted next-state residual
        self.out_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.eye_(self.out_proj.weight)  # identity init for stable rollouts

    def forward(self, z_history: torch.Tensor, a_tilde_t: torch.Tensor) -> torch.Tensor:
        """Predict next latent state from history + current action.
        z_history: (B, H_ctx, 192) — past H_ctx latent states
        a_tilde_t: (B, 192) — current action embedding
        Returns: (B, 192) — predicted residual for next state"""
        B = z_history.shape[0]

        # Append action as the (H_ctx+1)-th token
        tokens = torch.cat([z_history, a_tilde_t.unsqueeze(1)], dim=1)  # (B, H_ctx+1, 192)

        # Add positional embedding
        tokens = tokens + self.pos_emb[:, :tokens.shape[1], :]

        # Transformer forward
        out = self.transformer(tokens)  # (B, H_ctx+1, 192)

        # Use last token's output as predicted residual
        residual = self.out_proj(out[:, -1, :])  # (B, 192)
        return residual


class LeWMWorldModel:
    """Full LeWM engine: encoder + embedder + transformer predictor + CEM solver.
    Config per paper Section 8.1: embed_dim=192, history_size=3, predictor depth=6,
    heads=16, MLP dim=2048. CEM: num_samples=300, n_steps=30, topk=30."""

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 32,
        latent_dim: int = 192,
        history_size: int = 3,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_mlp_dim: int = 2048,
        cem_samples: int = 300,
        cem_iterations: int = 30,
        cem_topk: int = 30,
        horizon: int = 5,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.history_size = history_size
        self.predictor_depth = predictor_depth
        self.predictor_heads = predictor_heads
        self.predictor_mlp_dim = predictor_mlp_dim
        self.cem_samples = cem_samples
        self.cem_iterations = cem_iterations
        self.cem_topk = cem_topk
        self.horizon = horizon

        self.encoder = StateEncoder(state_dim, latent_dim)
        self.embedder = ActionEmbedder(action_dim, latent_dim)
        self.predictor = TransformerPredictor(
            latent_dim=latent_dim,
            depth=predictor_depth,
            heads=predictor_heads,
            mlp_dim=predictor_mlp_dim,
            history_size=history_size,
        )

        # Set to eval mode (random init per paper Section 9.4 limitation)
        self.encoder.eval()
        self.embedder.eval()
        self.predictor.eval()

    def encode_state(self, state_vector: List[float]) -> torch.Tensor:
        """Encode OS state vector into latent space."""
        s = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            z = self.encoder(s)
        return z.squeeze(0)

    def encode_action(self, action_vector: List[float]) -> torch.Tensor:
        """Encode action vector into latent space."""
        a = torch.tensor(action_vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a_tilde = self.embedder(a)
        return a_tilde.squeeze(0)

    def rollout(self, z_0: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """Autoregressive rollout for H steps with history context.
        z_history is a sliding window of H_ctx past states fed to the transformer."""
        z = z_0
        # Initialize history with copies of z_0 (padding)
        history = [z_0.clone() for _ in range(self.history_size)]

        for t in range(action_seq.shape[0]):
            a_tilde = self.embedder(action_seq[t].unsqueeze(0))
            # Build history tensor: (1, H_ctx, 192)
            z_hist = torch.stack(history[-self.history_size:], dim=0).unsqueeze(0)
            # Predict residual
            residual = self.predictor(z_hist, a_tilde)
            # z_{t+1} = 0.7 * z_t + 0.3 * residual
            z = 0.7 * z + 0.3 * residual.squeeze(0)
            # Update history
            history.append(z.clone())

        return z

    def batched_rollout(self, z_0: torch.Tensor, action_seqs: torch.Tensor) -> torch.Tensor:
        """Batched rollout: action_seqs shape (N, H, action_dim) → (N, latent_dim).
        Uses transformer predictor with history context."""
        N, H, A = action_seqs.shape
        # Embed all actions at once
        a_flat = action_seqs.reshape(N * H, A)
        a_tilde_all = self.embedder(a_flat)  # (N*H, 192)
        a_tilde_all = a_tilde_all.reshape(N, H, self.latent_dim)

        # Expand z_0 for N samples
        z = z_0.unsqueeze(0).expand(N, -1)  # (N, 192)

        # Initialize history: (N, H_ctx, 192) — all copies of z_0
        z_hist = z_0.unsqueeze(0).unsqueeze(0).expand(N, self.history_size, -1).clone()

        for t in range(H):
            a_t = a_tilde_all[:, t, :]  # (N, 192)
            # Predict residual via transformer
            residual = self.predictor(z_hist, a_t)  # (N, 192)
            # z_{t+1} = 0.7 * z_t + 0.3 * residual
            z = 0.7 * z + 0.3 * residual
            # Update history: shift left and append new z
            z_hist = torch.cat([z_hist[:, 1:, :], z.unsqueeze(1)], dim=1)

        return z

    def cem_plan(self, z_0: torch.Tensor, z_goal: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Cross-Entropy Method for latent space planning (vectorized).
        Returns (optimal_action_sequence, final_cost, elapsed_seconds)."""
        start = time.time()

        mu = torch.zeros(self.horizon * self.action_dim)
        sigma_sq = torch.ones(self.horizon * self.action_dim)

        best_cost = float('inf')
        best_seq = None

        with torch.no_grad():
            for iteration in range(self.cem_iterations):
                # Sample N candidates at once
                samples = torch.randn(self.cem_samples, self.horizon * self.action_dim) * \
                          torch.sqrt(sigma_sq).unsqueeze(0) + mu.unsqueeze(0)
                samples = samples.clamp(-3.0, 3.0)  # Clip to [-3, 3]

                # Reshape to (N, H, A) for batched rollout
                action_seqs = samples.reshape(self.cem_samples, self.horizon, self.action_dim)

                # Batched rollout: all N candidates in parallel
                z_finals = self.batched_rollout(z_0, action_seqs)  # (N, 192)

                # Compute costs: ||z_final - z_goal||^2 for all N
                diffs = z_finals - z_goal.unsqueeze(0)  # (N, 192)
                costs = (diffs ** 2).sum(dim=1)  # (N,)

                # Select top-K elites
                topk_costs, topk_indices = torch.topk(costs, self.cem_topk, largest=False)
                elites = samples[topk_indices]

                # Update distribution
                mu = elites.mean(dim=0)
                sigma_sq = torch.clamp(elites.var(dim=0), min=1e-4)

                if topk_costs[0].item() < best_cost:
                    best_cost = topk_costs[0].item()
                    best_seq = mu.reshape(self.horizon, self.action_dim)

        elapsed = time.time() - start
        return best_seq, best_cost, elapsed

    def validate_dag(
        self,
        state_vector: List[float],
        goal_vector: List[float],
        dag_actions: Optional[List[List[float]]] = None,
        cost_threshold: float = 50.0,
    ) -> dict:
        """Pre-execution validation: check if DAG action sequence reaches goal.
        Returns dict with cost, elapsed, valid flag."""
        z_0 = self.encode_state(state_vector)
        z_goal = self.encode_state(goal_vector)

        if dag_actions:
            # Use provided DAG actions
            action_tensor = torch.tensor(dag_actions, dtype=torch.float32)
            z_final = self.rollout(z_0, action_tensor)
            cost = torch.norm(z_final - z_goal).item() ** 2
            elapsed = 0.001  # negligible
        else:
            # Use CEM to find optimal path
            _, cost, elapsed = self.cem_plan(z_0, z_goal)

        return {
            "cost": round(cost, 4),
            "elapsed_s": round(elapsed, 4),
            "valid": cost < cost_threshold,
            "threshold": cost_threshold,
        }

    def proactive_simulate(
        self,
        state_vector: List[float],
        goal_vector: List[float],
        num_scenarios: int = 1000,
    ) -> dict:
        """Proactive simulation: generate scenarios and classify confidence.
        Per paper Section 4.3: runs 1000 scenarios, >95% confidence → automatic,
        60-80% → prompt user, <60% → hold."""
        z_0 = self.encode_state(state_vector)
        z_goal = self.encode_state(goal_vector)

        # Batch all scenarios: (num_scenarios, H, action_dim)
        with torch.no_grad():
            random_actions = torch.randn(num_scenarios, self.horizon, self.action_dim).clamp(-3, 3)
            z_finals = self.batched_rollout(z_0, random_actions)  # (N, 192)

            # Compute all costs at once
            diffs = z_finals - z_goal.unsqueeze(0)
            costs = (diffs ** 2).sum(dim=1)  # (N,)

        costs_arr = costs.numpy()
        confidence = float(np.mean(costs_arr < 50.0))

        if confidence > 0.95:
            action = "automatic"
        elif confidence > 0.60:
            action = "prompt_user"
        else:
            action = "hold"

        return {
            "num_scenarios": num_scenarios,
            "mean_cost": round(float(np.mean(costs_arr)), 4),
            "min_cost": round(float(np.min(costs_arr)), 4),
            "confidence": round(confidence, 4),
            "recommended_action": action,
        }
