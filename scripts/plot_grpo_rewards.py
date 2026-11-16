"""Plot mean reward and within-group reward standard deviation from GRPO logs."""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean

from formal_models import model_registry


ARMS = (
    ("baseline", "Baseline detector", "#285A92", "-"),
    ("arena", "Original Arena", "#A34222", "--"),
    ("mechanics", "Humor Mechanics", "#526820", ":"),
    ("adapted", "Adapted Arena", "#764B87", "-."),
)
METRICS = ("reward", "reward_std")
WINDOW = 10


def read_trace(path):
    state = json.loads(path.read_text(encoding="utf-8-sig"))
    by_step = {}
    for row in state["log_history"]:
        if not all(metric in row for metric in METRICS):
            continue
        step = int(row["step"])
        values = tuple(float(row[metric]) for metric in METRICS)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}: nonfinite reward at step {step}")
        by_step[step] = values
    if not by_step:
        raise ValueError(f"{path}: no training steps with reward and reward_std")
    return sorted(by_step.items())


def moving_mean(values):
    return [fmean(values[max(0, end - WINDOW):end]) for end in range(1, len(values) + 1)]


def plot(traces, scope, seed, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 5.4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharex=True)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.73, wspace=0.28)
    figure.suptitle(f"{scope} GRPO reward traces (seed {seed})", x=0.08, ha="left",
                    y=0.96, fontsize=18, fontweight="bold")

    for arm, label, color, style in ARMS:
        steps, values = zip(*traces[arm])
        for axis, metric in zip(axes, zip(*values)):
            axis.plot(steps, moving_mean(metric), label=label, color=color,
                      linestyle=style, linewidth=1.7)

    for axis, title in zip(axes, ("Mean training reward", "Within-group reward standard deviation")):
        axis.set_title(title, loc="left", fontweight="bold", fontsize=11)
        axis.set_xlabel("Optimization step")
        axis.grid(axis="y", color="#dbe1e8")
        axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 0.88),
                  ncol=4, frameon=False)
    figure.text(0.08, 0.065,
                "Trailing 10-step mean. Each run is scored by its own reward model.",
                fontsize=9, color="#46556b")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", nargs="?", default="member_1",
                        choices=("member_1", "member_2", "member_3", "member_4", "team"))
    parser.add_argument("--llm", choices=model_registry(), default="qwen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-root", type=Path, default=Path("results/formal_v2"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot_scope = args.scope if args.llm == "qwen" else f"{args.scope}_{args.llm}"
    output = args.output or (
        args.run_root / "plots" / f"{plot_scope}_grpo_reward_traces_seed{args.seed}.png"
    )
    try:
        traces = {
            arm: read_trace(
                args.run_root / f"{args.scope}_{args.llm}_grpo_{arm}_seed{args.seed}" / "trainer_state.json"
            )
            for arm, *_ in ARMS
        }
        label = "Team" if args.scope == "team" else f"Member {args.scope.rsplit('_', 1)[1]}"
        if args.llm != "qwen":
            label = f"{label} / {args.llm}"
        plot(traces, label, args.seed, output)
    except ModuleNotFoundError as error:
        if error.name == "matplotlib":
            parser.exit(1, "Install Matplotlib first: uv pip install --python .venv/bin/python matplotlib\n")
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
