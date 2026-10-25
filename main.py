import argparse
import json
import sys

from src.config import PROJECT, absolute, preflight, qa_dataset, select_detector, select_llm
from src.data import normalize, prepare_dataset, read_json, write_json


def llm_arguments(p, adapter=False):
    p.add_argument("--llm", default="qwen", help="Name in configs/llms.json")
    p.add_argument("--model", help="Override model folder, cached HF repo ID, or Ollama model name")
    p.add_argument("--backend", choices=["transformers", "ollama"],
                   help="Default configured backend: Hugging Face Transformers. Ollama is optional.")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--seed", type=int, default=42)
    if adapter:
        p.add_argument("--adapter", help="Optional LoRA adapter folder")


def parser():
    p = argparse.ArgumentParser(description="Generate sentences, rank for review, select one per product, train and test LoRA")
    commands = p.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="List model choices and locations")
    models.add_argument("--detector-root")
    products = commands.add_parser("products", help="Extract product descriptions from the detector QA dataset")
    products.add_argument("--csv", help="QA CSV override; defaults to configs/datasets.json")
    products.add_argument("--dataset-root", help="Root containing processed/unbiased/train.csv")
    products.add_argument("--limit", type=int)
    products.add_argument("--output", default="data/products_from_qa.json")
    products.add_argument("--seed", type=int, default=42)
    generate = commands.add_parser("generate", help="Generate three sentences per product")
    llm_arguments(generate, adapter=True)
    generate.add_argument("--products", help="Optional extracted products JSON; otherwise use the QA training CSV")
    generate.add_argument("--dataset-root")
    generate.add_argument("--num-products", type=int, default=50, help="Product count sampled from the QA CSV")
    generate.add_argument("--per-product", type=int, default=3)
    generate.add_argument("--output", default="results/sentences.json")
    rank = commands.add_parser("rank", help="Detector scoring and Excel export")
    rank.add_argument("--input", default="results/sentences.json")
    rank.add_argument("--output", default="results/ranked.xlsx")
    rank.add_argument("--detector", default="roberta")
    rank.add_argument("--detector-root")
    rank.add_argument("--batch-size", type=int, default=16)
    rank.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    rank.add_argument("--top-k", type=int, default=3)
    rank.add_argument("--check", action="store_true", help="Check selected detector files without loading GPU weights")
    rank.add_argument("--seed", type=int, help="Optional checkpoint seed; uses model default if omitted")
    review = commands.add_parser("review", help="Use an LLM to select one of three sentences per product")
    llm_arguments(review, adapter=True)
    review.add_argument("--input", default="results/ranked.xlsx")
    review.add_argument("--output", default="results/reviewed.xlsx")
    review.add_argument("--overwrite-selection", action="store_true")
    prepare = commands.add_parser("prepare", help="Create LoRA data from selected=1 rows")
    prepare.add_argument("--input", default="results/reviewed.xlsx")
    prepare.add_argument("--output", default="data/lora")
    prepare.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train", help="Train a new LoRA adapter")
    llm_arguments(train)
    train.add_argument("--data", default="data/lora")
    train.add_argument("--config", default="configs/training.json")
    train.add_argument("--output", help="Default: results/LLM_lora")
    test = commands.add_parser("test", help="Compare base model and adapter on held-out test products")
    llm_arguments(test)
    test.add_argument("--adapter", required=True)
    test.add_argument("--data", default="data/lora/test.jsonl")
    test.add_argument("--output", default="results/test.xlsx")
    test.add_argument("--limit", type=int)
    inspect = commands.add_parser("inspect-detector", help="Show safe checkpoint metadata for a custom model loader")
    inspect.add_argument("--detector", required=True)
    inspect.add_argument("--detector-root")
    inspect.add_argument("--seed", type=int)
    return p


def new_output(path):
    path = absolute(path)
    if path.exists():
        raise FileExistsError(f"Output exists; choose a new path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == "models":
        print("LLMs:")
        for name in read_json(PROJECT / "configs/llms.json"):
            config = select_llm(name)
            print(f"  {name:22} {config['model']}")
        print("Detectors:")
        for name in read_json(PROJECT / "configs/detectors.json"):
            config = select_detector(name, root=args.detector_root)
            status = "loader required" if config["backend"] == "custom" and not config.get("factory") else "configured"
            print(f"  {name:22} {status:16} {config['checkpoint']}")
        return
    if args.command == "products":
        from src.products import from_qa_csv
        output = new_output(args.output)
        rows = from_qa_csv(qa_dataset(args.csv, args.dataset_root), args.limit, args.seed)
        write_json(output, rows)
        print(f"Products: {len(rows)} -> {output}")
    elif args.command == "generate":
        if args.per_product < 1:
            p.error("--per-product must be positive.")
        from src.llm import LanguageModel, generate
        output = new_output(args.output)
        config = select_llm(args.llm, args.backend, args.model, args.adapter)
        if args.products:
            products = read_json(absolute(args.products))
        else:
            from src.products import from_qa_csv
            products = from_qa_csv(qa_dataset(root=args.dataset_root), args.num_products, args.seed)
        model = LanguageModel(config, args.ollama_url, args.seed)
        write_json(output, generate(model, products, args.per_product))
        print(f"Sentences: {output}")
    elif args.command == "rank":
        if args.batch_size < 1 or args.top_k < 1:
            p.error("--batch-size and --top-k must be positive.")
        config = preflight(select_detector(args.detector, args.seed, args.detector_root))
        if args.check:
            print(json.dumps(config, indent=2))
            return
        from src.detector import create_detector
        from src.review import create_excel, rank_rows
        output = new_output(args.output)
        rows = normalize(read_json(absolute(args.input)))
        detector = create_detector(config, args.device, args.batch_size)
        create_excel(rank_rows(rows, detector.score(rows)), output, args.top_k)
        print(f"Ranking: {output}")
    elif args.command == "review":
        from src.llm import LanguageModel
        from src.review import choose_with_llm, read_review, save_choices
        output = new_output(args.output)
        source = absolute(args.input)
        rows = read_review(source)
        model = LanguageModel(select_llm(args.llm, args.backend, args.model, args.adapter), args.ollama_url, args.seed)
        save_choices(source, output, choose_with_llm(rows, model, args.overwrite_selection))
        print(f"Selections: {output}")
    elif args.command == "prepare":
        from src.review import read_review, select_training_rows
        output = absolute(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        counts = prepare_dataset(select_training_rows(read_review(absolute(args.input))), output, args.seed)
        print(f"LoRA data: {output} {counts}")
    elif args.command == "train":
        if args.backend == "ollama":
            p.error("Training requires a local Transformers model.")
        from src.training import train
        config = read_json(absolute(args.config))
        config.update(select_llm(args.llm, "transformers", args.model))
        config["seed"] = args.seed
        output = new_output(args.output or f"results/{args.llm}_lora")
        data = absolute(args.data)
        for split in ["train", "validation", "test"]:
            if not (data / f"{split}.jsonl").is_file():
                raise FileNotFoundError(data / f"{split}.jsonl")
        train(config, data, output)
        write_json(output / "training_config.json", config)
        print(f"Adapter: {output / 'final_adapter'}")
    elif args.command == "test":
        if args.backend == "ollama":
            p.error("Adapter comparison requires a Transformers model.")
        if args.limit is not None and args.limit < 1:
            p.error("--limit must be positive.")
        from src.testing import compare
        compare(select_llm(args.llm, "transformers", args.model), absolute(args.adapter),
                absolute(args.data), new_output(args.output), args.limit, args.seed)
        print(f"Comparison: {absolute(args.output)}")
    elif args.command == "inspect-detector":
        import torch
        config = select_detector(args.detector, args.seed, args.detector_root)
        path = absolute(config["checkpoint"])
        if path.is_dir():
            path = path / "best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
        print(json.dumps({"path": str(path), "keys": list(checkpoint), "config": checkpoint.get("config"),
                          "vocab_size": len(checkpoint.get("vocab", {})),
                          "state_shapes": {k: list(v.shape) for k, v in state.items() if hasattr(v, "shape")}},
                         indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
