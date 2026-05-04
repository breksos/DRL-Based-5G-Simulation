# Training Notes

Actual episodes trained per scenario (read from CSV logs):

| Scenario | Config n_episodes | Actually trained | Notes |
|---|---|---|---|
| A | 3676 | 3676 | Training stopped early (not at the 10k config target) |
| B | 2000 | 2000 | — |
| C | 2000 | 2000 | — |
| D | 2000 | 2000 | — |

Configs (`configs/scenario_*.yaml`) have been updated to reflect the actually-trained
episode count so that future re-runs reproduce the same checkpoints.

The original config had `n_episodes: 10000` for all scenarios; this was the intended
budget but training was interrupted or stopped early for B, C, D and crashed/stopped
at 3676 for A. Do not re-train without first reviewing resource costs.

**Reward-scale mismatch:** Scenario A was retrained (500 ep) after `env._compute_reward`
was updated to normalise throughput by `THROUGHPUT_NORM_MBPS = 50.0`. Checkpoints for
scenarios B, C, and D (`model_final.pth`) predate this change and were trained on the
old scale (throughput / 1e6 Mbps, unnormalised). Loading them for continued training
will produce corrupted Q-value updates; pure inference via `DQNScheduler` is unaffected.
