import csv

from src.config import absolute, select_detector
from src.data import sha256, write_json
from .data import classification_metrics, read_examples, split_path
from .inference import create_detector


def evaluate_detector(args):
    config = select_detector(args.detector, args.seed)
    source = split_path("test", args.csv)
    rows = read_examples(source, config.get("input_mode", "pair"))
    output = absolute(args.output)
    if output.exists():
        raise FileExistsError(output)
    detector = create_detector(config, args.device, args.batch_size)
    scores = list(detector.score(rows))
    from src.data import score_rows
    score_rows(rows, scores, .5)  # validate the backend's probability contract
    positive = config["positive_label"]
    predicted = [positive if score >= .5 else 1-positive for score in scores]
    metrics = classification_metrics([r["label"] for r in rows], predicted)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", {"detector": config, "data_sha256": sha256(source), **metrics})
    with (output / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_index", "actual", "predicted", "humor_probability"])
        writer.writerows((i, row["label"], prediction, score)
                         for i, (row, prediction, score) in enumerate(zip(rows, predicted, scores)))
    print(f"Accuracy: {metrics['accuracy']:.4f}; macro-F1: {metrics['macro_f1']:.4f}", flush=True)
    print(f"Evaluation: {output}", flush=True)
