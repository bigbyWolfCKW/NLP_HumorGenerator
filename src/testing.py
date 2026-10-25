from pathlib import Path

def compare(llm_config, adapter, test_path, output, limit=None, seed=42):
    import json
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from .llm import LanguageModel
    from .artifacts import check_weights
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    check_weights(Path(adapter) / "adapter_model.safetensors")
    examples = [json.loads(line) for line in Path(test_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        examples = examples[:limit]
    if not examples:
        raise ValueError("Test dataset is empty.")
    llm_config = dict(llm_config, backend="transformers", adapter=str(adapter))
    llm = LanguageModel(llm_config, seed=seed)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Test_comparison"
    sheet.append(["prompt", "reference", "base_model", "lora_model"])
    for i, example in enumerate(examples, 1):
        messages = example["messages"]
        if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError("Test examples must contain a user prompt and assistant reference.")
        prompt, reference = messages[0]["content"], messages[1]["content"]
        with llm.model.disable_adapter():
            baseline = llm.answer(prompt)
        tuned = llm.answer(prompt)
        sheet.append([prompt, reference, baseline, tuned])
        print(f"Compared {i}/{len(examples)}", flush=True)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    for column in "ABCD":
        sheet.column_dimensions[column].width = 65
    for row in sheet:
        sheet.row_dimensions[row[0].row].height = 85 if row[0].row > 1 else 25
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.font = Font(name="Arial", size=10, bold=cell.row == 1)
            if isinstance(cell.value, str):
                cell.data_type = "s"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        workbook.save(stream)
    workbook.close()
