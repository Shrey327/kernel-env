"""Prepare allowlisted public Hugging Face artifacts; never upload from the working tree directly."""
import hashlib
import json
import runpy
import shutil
import struct
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs/proof-20261019/results"
OUT = ROOT / "runs/proof-20261019/huggingface"
SPACE = OUT / "space"
MODEL = OUT / "model"
REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
ADAPTER_SHA = "23fb8fe28a8f91b925e6cff1f802f14e802515f80382b92204563070d280e6fe"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare():
    assert not OUT.exists(), f"Preserve the existing release: {OUT}"
    original = RESULTS / "adapter/adapter_model.safetensors"
    assert digest(original) == ADAPTER_SHA, "Unexpected adapter weights"
    with original.open("rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    assert len(set(header) - {"__metadata__"}) == 392
    for target in (SPACE, MODEL / "evaluation"):
        target.mkdir(parents=True)
    for folder in ("kernel_env", "train"):
        for src in (ROOT / folder).rglob("*"):
            rel = src.relative_to(ROOT)
            if (not src.is_file() or src.is_symlink() or
                    any(part.startswith(".") or part == "__pycache__" for part in rel.parts) or
                    src.suffix not in {".py", ".md", ".toml", ".yaml", ".json"}):
                continue
            dst = SPACE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    shutil.copyfile(ROOT / "modal_app.py", SPACE / "modal_app.py")
    for src, dst in (("SPACE_README.md", SPACE / "README.md"),
                     ("MODEL_README.md", MODEL / "README.md"),
                     ("Dockerfile", SPACE / "Dockerfile"),
                     ("index.html", SPACE / "index.html"),
                     ("client_smoke.py", SPACE / "examples/client_smoke.py")):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "hf_release" / src, dst)
    # The fixture stores source strings; this does not import or execute generated kernels.
    values = runpy.run_path(str(ROOT / "kernel_env/tests/kernels.py"))
    (SPACE / "examples/relu.py").write_text(values["GOOD"])
    (SPACE / ".dockerignore").write_text(".git\n__pycache__\n*.pyc\n.venv\ntrain\nevaluation\n")
    licenses = ROOT / "runs/proof-20261019/hf-licenses"
    shutil.copyfile(licenses / "OPENENV-LICENSE", SPACE / "LICENSE")
    shutil.copyfile(licenses / "MODEL-LICENSE", MODEL / "LICENSE")
    shutil.copyfile(original, MODEL / original.name)
    config = json.loads((RESULTS / "adapter/adapter_config.json").read_text())
    config["revision"] = REVISION
    (MODEL / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ("comparison.json", "manifest.json", "signed_validation.json", "gelu_after.py"):
        shutil.copyfile(RESULTS / name, MODEL / "evaluation" / name)
    provenance = json.loads((RESULTS / "provenance.json").read_text())
    provenance.pop("gpus")  # Device UUIDs are unnecessary for a public release.
    provenance["hardware"] = "2 x NVIDIA H100 80GB HBM3; separate model/evaluator devices"
    (MODEL / "evaluation/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    state = json.loads((RESULTS / "adapter/checkpoint-16/trainer_state.json").read_text())
    metrics = [row for row in state["log_history"] if "loss" in row]
    assert len(metrics) == 16 and all(row["grad_norm"] > 0 for row in metrics)
    (MODEL / "evaluation/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    release = {"date": "2026-10-19", "experiment": "proof-20261019-7b", "step": 16,
               "adapter_sha256": ADAPTER_SHA, "base_revision": REVISION,
               "publication_config_change": "revision pinned; adapter tensors unchanged",
               "space": "Shreyansh327/kernel-env", "hosting": "static; no GPU execution",
               "files_sha256": {str(p.relative_to(MODEL)): digest(p)
                                  for p in sorted(MODEL.rglob("*")) if p.is_file()}}
    (MODEL / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    with tarfile.open(SPACE / "kernel-env-source.tar.gz", "w:gz") as tar:
        for name in ("kernel_env", "train", "modal_app.py", "Dockerfile", "LICENSE", "README.md", "examples"):
            tar.add(SPACE / name, arcname="kernel-env/" + name)
    (SPACE / "release.json").write_text(json.dumps({"date": "2026-10-19", "hosting": "static",
        "source_archive_sha256": digest(SPACE / "kernel-env-source.tar.gz"),
        "source_sha256": {str(p.relative_to(SPACE)): digest(p) for p in sorted(SPACE.rglob("*.py"))}}, indent=2))
    print(json.dumps({"space": str(SPACE), "model": str(MODEL), "adapter_sha256": ADAPTER_SHA}))


if __name__ == "__main__":
    prepare()
