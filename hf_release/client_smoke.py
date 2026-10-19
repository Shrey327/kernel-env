"""Submit the included real Triton example to a running GPU environment."""
import argparse
from pathlib import Path

from kernel_env import KernelAction, KernelEnv

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
args = parser.parse_args()
with KernelEnv(base_url=args.url).sync() as env:
    initial = env.reset(seed=42, task_id="level1/19_ReLU")
    Path("task_prompt.txt").write_text(initial.observation.problem)
    source = Path(__file__).with_name("relu.py").read_text()
    result = env.step(KernelAction(code=source, language="triton"))
    print(result.observation.model_dump_json(indent=2))
    if result.observation.correct is not True or "[mock]" in result.observation.feedback:
        raise SystemExit("Expected a correct result from the real GPU evaluator")
