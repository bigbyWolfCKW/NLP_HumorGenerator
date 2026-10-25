import json
import struct
import zipfile
from pathlib import Path


def check_weights(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.suffix == ".safetensors":
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 100_000_000:
                raise ValueError(f"Invalid weights header: {path}")
            header = json.loads(stream.read(length))
        expected = 8 + length + max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        if path.stat().st_size != expected:
            raise ValueError(f"Incomplete weights: {path} (expected {expected} bytes)")
    elif path.suffix == ".pt" and not zipfile.is_zipfile(path):
        raise ValueError(f"Invalid PyTorch checkpoint: {path}")


def check_hf_weights(folder):
    folder = Path(folder)
    for name in ["model.safetensors", "pytorch_model.bin"]:
        if (folder / name).is_file():
            check_weights(folder / name)
            return
    for name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        if (folder / name).is_file():
            data = json.loads((folder / name).read_text(encoding="utf-8"))
            for shard in set(data["weight_map"].values()):
                check_weights(folder / shard)
            return
    raise FileNotFoundError(f"No model weights in {folder}")
