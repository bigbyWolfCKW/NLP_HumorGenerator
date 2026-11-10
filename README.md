# ProductPunch

ProductPunch trains and evaluates Qwen models that write humorous English
ads grounded in product facts. Each team member trains on their assigned
products and compares six model variants on the same held-out products.

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

See the [full workflow](docs/formal-workflow.md) for input paths,
checkpoint paths, and direct Python commands.

## Run one stage at a time

| Stage | Command | When to run |
| --- | --- | --- |
| Generate training ads | `sbatch formal.sbatch generate-ads` | Start here if you need a new candidate run. To reuse a compatible completed run, set `CANDIDATE_RUN` to its directory. |
| Prepare SFT data | `bash formal.sbatch prepare-sft` | Save the AI's `reviewed_candidates.csv` in the candidate run directory first. |
| Export blind reward pairs | `bash formal.sbatch prepare-humor-pairs` | Run after candidate generation. Give the reviewer `blind_pairs.jsonl` and the reward-label prompt. |
| Import reward labels | `bash formal.sbatch prepare-reward-data` | Save the AI's `reviewed_pairs.jsonl` beside the blind pairs. The importer uses the matching private mapping from that directory. |
| Train SFT | `sbatch formal.sbatch train-sft` | Run when the reviewed SFT JSONL is ready. |
| Train the adapted reward | `sbatch formal.sbatch train-reward` | Run when the reward train and validation JSONLs are ready. This can run in parallel with SFT. |
| Train GRPO | `sbatch formal.sbatch train-grpo <reward>` | Run once per reward with the same SFT adapter. The `adapted` arm also needs the adapted reward checkpoint. |
| Generate held-out ads | `sbatch formal.sbatch evaluate <variant>` | Run each variant after its required model or adapter is ready. |
| Export final blind pairs | `bash formal.sbatch prepare-final-pairs` | Run after both selected evaluation runs finish. By default this compares original Arena with adapted Arena. |

Submit the next stage once its inputs are ready. At an AI review
handoff, save the returned file in the specified directory before
running the import stage.

## Six held-out comparisons

Use the same test products, prompt, seed, and number of candidates
for every variant.

| Evaluation variant | Model | GRPO reward argument |
| --- | --- | --- |
| `base` | Qwen2.5-7B-Instruct | — |
| `sft_pooled` | Qwen + SFT | — |
| `grpo_baseline` | Qwen + SFT + GRPO with the binary detector | `baseline` |
| `grpo_original_arena` | Qwen + SFT + GRPO with original Arena | `arena` |
| `grpo_mechanics` | Qwen + SFT + GRPO with Humor Mechanics | `mechanics` |
| `grpo_adapted` | Qwen + SFT + GRPO with adapted Arena | `adapted` |

Use the [review prompts](docs/formal-review-prompts.md) for SFT selection,
reward labels, and final held-out evaluation. The workflow guide also
describes the optional six-model scorecard. Keep final held-out
judgments separate from training data.

## Remarks

- **Default AI review.** Manually give the external AI the review input
  and the matching prompt, then save its response in the expected
  location. If you use an AI web app on your computer, download the
  review input from SuperPOD and upload the returned file afterward.
  For blind reviews, share only the public review input and the prompt.
  Keep the private mapping with the run owner.
- **Optional automation.** The earlier pipeline's API reviewer is
  configured in `test.sbatch` and can serve as a starting point.
  Automating a formal review stage requires the corresponding output
  format and the original candidate or pair IDs. Store any API key in
  a private environment variable or `.env`.
- **GitHub handoff.** Share the scripts, prompts, and documentation in
  Git. Transfer assigned datasets and required checkpoints separately.

See the [archived README](README.legacy.md) for the earlier
`test.sbatch` workflow.