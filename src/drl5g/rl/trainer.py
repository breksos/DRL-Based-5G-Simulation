"""Training loop for the DQN scheduler."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from drl5g.channel import capacity_bps, drop_ues, snr_db
from drl5g.config import ScenarioConfig
from drl5g.rl.allocation import RBGAllocationTask
from drl5g.rl.dqn import DQNAgent, EquivariantQNetwork
from drl5g.schedulers.dqn import CHECKPOINT_FORMAT

log = logging.getLogger(__name__)

VALIDATION_DROPS = 128
VALIDATE_EVERY = 100
LOG_FIELDS = ("episode", "train_objective", "epsilon", "mean_loss", "validation_objective")

# Independent random streams per purpose, so training never sees an
# evaluation drop: evaluation uses np.random.default_rng(seed) directly.
_TRAIN_STREAM, _VALIDATION_STREAM = 1, 2


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    log: Path
    best_validation_objective: float
    best_episode: int
    seconds: float


def validation_capacities(config: ScenarioConfig) -> np.ndarray:
    """A fixed set of UE drops used to track progress and pick the best weights."""
    rng = np.random.default_rng([config.train.seed, _VALIDATION_STREAM])
    drops = [
        drop_ues(config.env.n_ue, config.env.cell_radius, rng)[0] for _ in range(VALIDATION_DROPS)
    ]
    return capacity_bps(snr_db(np.stack(drops)))


@torch.no_grad()
def greedy_objective(
    q_net: EquivariantQNetwork, task: RBGAllocationTask, capacities: np.ndarray
) -> float:
    """Mean objective reached by the greedy policy of ``q_net`` over ``capacities``."""
    total = 0.0
    for capacity in capacities:
        obs = task.reset(capacity)
        while not task.done:
            q_values = q_net(torch.from_numpy(obs).unsqueeze(0))
            obs, _, _ = task.step(int(q_values.argmax()))
        total += task.objective
    return total / len(capacities)


def save_checkpoint(
    path: Path, q_net: EquivariantQNetwork, config: ScenarioConfig, **metadata: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "config": config.to_dict(),
            "q_net": q_net.state_dict(),
            **metadata,
        },
        path,
    )


def train(
    config: ScenarioConfig,
    checkpoint_path: Path,
    log_path: Path,
    episodes: int | None = None,
) -> TrainingResult:
    """Train a DQN scheduler for ``config`` and save the best validated weights.

    One episode is one TTI built decision by decision on a fresh UE drop.
    Every ``VALIDATE_EVERY`` episodes the greedy policy is scored on a fixed
    validation set; the checkpoint keeps the best-scoring weights.
    """
    n_episodes = episodes or config.train.episodes
    seed = config.train.seed
    task = RBGAllocationTask(
        config.env, config.dqn.rbg_size, rng=np.random.default_rng([seed, _TRAIN_STREAM])
    )
    agent = DQNAgent(
        obs_size=task.observation_size,
        n_actions=config.env.n_ue,
        config=config.dqn,
        total_steps=n_episodes * task.n_decisions,
        seed=seed,
    )
    eval_task = RBGAllocationTask(config.env, config.dqn.rbg_size)
    validation = validation_capacities(config)

    best_score, best_episode = -np.inf, 0
    best_weights = {k: v.clone() for k, v in agent.q_net.state_dict().items()}
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(LOG_FIELDS)
        for episode in range(1, n_episodes + 1):
            obs = task.reset()
            losses = []
            while not task.done:
                action = agent.act(obs)
                next_obs, reward, done = task.step(action)
                loss = agent.observe(obs, action, reward, next_obs, done)
                if loss is not None:
                    losses.append(loss)
                obs = next_obs

            validation_score = None
            if episode % VALIDATE_EVERY == 0 or episode == n_episodes:
                validation_score = greedy_objective(agent.q_net, eval_task, validation)
                if validation_score > best_score:
                    best_score, best_episode = validation_score, episode
                    best_weights = {k: v.clone() for k, v in agent.q_net.state_dict().items()}
                log.info(
                    "episode %5d/%d  ε=%.3f  validation objective %.4f (best %.4f @ %d)",
                    episode,
                    n_episodes,
                    agent.epsilon,
                    validation_score,
                    best_score,
                    best_episode,
                )

            writer.writerow(
                [
                    episode,
                    round(task.objective, 5),
                    round(agent.epsilon, 4),
                    round(float(np.mean(losses)), 6) if losses else "",
                    round(validation_score, 5) if validation_score is not None else "",
                ]
            )

    agent.q_net.load_state_dict(best_weights)
    save_checkpoint(
        checkpoint_path,
        agent.q_net,
        config,
        episodes=n_episodes,
        best_episode=best_episode,
        validation_objective=float(best_score),
    )
    seconds = time.perf_counter() - start
    log.info("saved %s (validation objective %.4f) in %.0f s", checkpoint_path, best_score, seconds)
    return TrainingResult(checkpoint_path, log_path, float(best_score), best_episode, seconds)
