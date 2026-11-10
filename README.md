# ProductPunch

ProductPunch trains and evaluates Qwen models that write humorous English
ads grounded in product facts. Each team member trains on their assigned
products and evaluates six model variants on the same held-out products.

## Quick start on SuperPOD

From the repository root, set up the environment and select your assigned
member ID.

```bash
source setup.sh
module load slurm
export MEMBER_ID=member_2  # Replace with your assigned ID.
```

Before submitting jobs, make the local Qwen2.5-7B-Instruct model and the
required reward checkpoints available. Use the train and test CSVs and
`assignment_core.json` from the same team assignment. Check the Slurm
account in `formal.sbatch` against your allocation.

See the [formal workflow guide](docs/formal-workflow.md) for input formats,
checkpoint requirements, and direct Python commands. Use the defaults
below when running through `formal.sbatch`.

## Run one stage at a time

| Stage | Command | When to run |
| --- | --- | --- |
| Generate training ads | `sbatch formal.sbatch generate-ads` | Start here if you need a new candidate run. To reuse a compatible completed run, set `CANDIDATE_RUN` to its directory. |
| Prepare SFT data | `bash formal.sbatch prepare-sft` | Save the AI's `reviewed_candidates.csv` in the candidate run directory first. |
| Export blind reward pairs | `bash formal.sbatch prepare-humor-pairs` | Run after candidate generation. Give the reviewer `blind_pairs.jsonl` and the reward-label prompt. |
| Import reward labels | `bash formal.sbatch prepare-reward-data` | Save the AI's `reviewed_pairs.jsonl` beside the blind pairs. The importer uses the matching private mapping from that directory. |
| Train SFT | `sbatch formal.sbatch train-sft` | Run when the reviewed SFT JSONL is ready. |
| Train the adapted reward | `sbatch formal.sbatch train-reward` | Run when the reward train and validation JSONLs are ready. This can run in parallel with SFT. |
| Train GRPO | `sbatch formal.sbatch train-grpo <reward>` | Run once per reward with the same SFT adapter. The `adapted` run also needs the adapted reward checkpoint. |
| Generate held-out ads | `sbatch formal.sbatch evaluate <variant>` | Run each variant after its required model or adapter is ready. |
| Export final blind pairs | `bash formal.sbatch prepare-final-pairs` | Run after both selected evaluation runs finish. The default comparison is original Arena versus adapted Arena. |

### Default reward paths

| Artifact | Path used by `formal.sbatch` |
| --- | --- |
| Blind pairs and returned reviews | `data/formal_v2/reward/${MEMBER_ID}/humor_pairwise/` |
| Imported reward training and validation data | `data/formal_v2/reward/${MEMBER_ID}/reward_data_review/` |
| Adapted reward model | `models/detector/preferences/${MEMBER_ID}_product_humor_review/best_model/` |

If you use the direct commands in the guide for any reward stage, match
their paths to the table above or set `REWARD_DATA` and `REWARD_MODEL`
consistently across runs. `REWARD_MODEL` points to the parent directory
that contains `best_model/`.

## Six held-out model variants

Use the same test products, prompt, seed, and number of candidates for
every variant.

| Evaluation variant | Model | GRPO reward argument |
| --- | --- | --- |
| `base` | Qwen2.5-7B-Instruct | — |
| `sft_pooled` | Qwen + SFT | — |
| `grpo_baseline` | Qwen + SFT + GRPO with the binary detector | `baseline` |
| `grpo_original_arena` | Qwen + SFT + GRPO with original Arena | `arena` |
| `grpo_mechanics` | Qwen + SFT + GRPO with Humor Mechanics | `mechanics` |
| `grpo_adapted` | Qwen + SFT + GRPO with adapted Arena | `adapted` |

Evaluation outputs are saved under
`results/formal_v2/evaluation/${MEMBER_ID}/`. After generating the ads,
use the [held-out review prompt](docs/formal-review-prompts.md#final-held-out-evaluation)
with the exported blind pairs. Keep final held-out judgments separate
from training data.

## Remarks

- **Default AI review.** Manually give the external AI the review input
  and the [matching prompt](docs/formal-review-prompts.md). Save the
  response at the path expected by the next stage. If you use an AI web
  app on your computer, download the input from SuperPOD and upload the
  response afterward. For blind reviews, share only the public review
  input and prompt. Keep the private mapping with the run owner.
- **Optional automation.** The earlier pipeline's API reviewer in
  `test.sbatch` can serve as a starting point. Automation for a formal
  review stage must return the required file format and preserve the
  original candidate or pair IDs. Store credentials in a private
  environment variable or `.env`.
- **GitHub handoff.** Share scripts, prompts, and documentation in Git.
  Transfer assigned datasets and required checkpoints separately.

See the [archived README](README.legacy.md) for the earlier
`test.sbatch` workflow.