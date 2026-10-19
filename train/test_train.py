"""Mac-runnable checks. Rollout test needs a mock server:

    cd ../kernel_env && KERNEL_ENV_EVALUATOR=mock KERNELBENCH_DIR=<dir with level1/*.py> \
        .venv/bin/uvicorn server.app:app --port 8765
    KERNEL_ENV_URL=http://127.0.0.1:8765 KERNEL_ENV_TASK=level1/19_ReLU .venv/bin/python -m pytest test_train.py
"""

import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from baseline_eval import extract_code, run_episodes, run_episodes_pipelined, summarize
from grpo_kernel import KernelArgs, episode_return, feedback_ids, make_rollout_func

sys.path.insert(0, str(Path(__file__).parent.parent / "kernel_env"))
from tests.kernels import GOOD  # noqa: E402  (kernel_env's fixture kernel)

URL = os.environ.get("KERNEL_ENV_URL")
TASK = os.environ.get("KERNEL_ENV_TASK", "level1/19_ReLU")
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
BAD = "class ModelNew:\n    pass"  # no kernel -> rejected by static checks, reward 0


def test_extract_code():
    assert extract_code("hi\n```python\na = 1\n```\nbye") == "a = 1"
    assert extract_code("```py\nx\n```\n```python\nlast\n```") == "last"
    assert extract_code("```\nplain\n```") == "plain"
    assert extract_code("```python\nb = 2\n```\n```text\nlog\n```") == "b = 2"  # python beats later non-python
    assert extract_code("```python\ncut = 1\nmore") == "cut = 1\nmore"  # truncated fence
    assert extract_code("  import torch\n") == "import torch"  # bare source, as the prompt asks


def test_episode_return():
    assert episode_return([0.0, 0.3, 1.3], "discounted", 0.5) == pytest.approx(0.5 * 0.3 + 0.25 * 1.3)
    assert episode_return([0.0, 1.3, 0.3], "best", 0.5) == 1.3
    assert episode_return([], "best", 1.0) == 0.0


def test_summarize():
    eps = [{"turns": [{"stage": "format", "correct": False, "speedup": 0.0},
                      {"stage": "timed", "correct": True, "speedup": 2.0}]},
           {"turns": [{"stage": "compile", "correct": False, "speedup": 0.0}]}]
    s = summarize(eps)
    assert (s["compile_pct"], s["correct_pct"], s["correct_turn1_pct"], s["median_speedup"]) == (50, 50, 0, 2.0)
    assert s["stage_hist_furthest"] == {"timed": 1, "compile": 1}


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER)


def test_feedback_ids_stitch_like_chat_template():
    tok = _tok()
    conv = [{"role": "user", "content": "P"}, {"role": "assistant", "content": "A1"}, {"role": "user", "content": "F"}]
    want = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=False)
    p = tok.apply_chat_template(conv[:1], add_generation_prompt=True, tokenize=True, return_dict=False)
    got = p + tok.encode("A1") + [tok.eos_token_id] + feedback_ids(tok, "F")
    assert got == want


class FakeTrainer:
    """Stands in for GRPOTrainer: replies BAD on turn 1 and GOOD (fenced) afterwards."""

    def __init__(self, tok):
        self.processing_class, self.tok, self.calls = tok, tok, 0
        self.model = SimpleNamespace(training=True)
        self.state = SimpleNamespace(global_step=0)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _generate_single_turn(self, prompt_ids, images, mm, n):
        self.calls += 1
        text = BAD if self.calls == 1 else f"Here:\n```python\n{GOOD}\n```"
        ids = self.tok.encode(text) + [self.tok.eos_token_id]
        return [ids for _ in prompt_ids], None


@pytest.mark.skipif(not URL, reason="set KERNEL_ENV_URL to a mock kernel_env server")
def test_rollout_func_against_mock_server(tmp_path):
    tok, n = _tok(), 3
    trainer = FakeTrainer(tok)
    fn = make_rollout_func(KernelArgs(env_url=URL, gamma=0.5), str(tmp_path / "s.jsonl"))
    out = fn([TASK] * n, trainer)

    assert out["logprobs"] is None and len(out["prompt_ids"]) == n
    max_turns = trainer.calls  # one generate call per turn, all episodes in lockstep
    for p, c, m in zip(out["prompt_ids"], out["completion_ids"], out["env_mask"]):
        assert len(c) == len(m)
        # stitched token stream == chat template of the same conversation
        text = tok.decode(p + c)
        assert text.count("<|im_start|>assistant") == max_turns and "stage=format" in text
        model_text = tok.decode([t for t, k in zip(c, m) if k])
        assert BAD in model_text and "stage=" not in model_text  # feedback tokens are masked out
    # mock evaluator: turn 1 rejected (0), later turns speedup 1.0 -> 0.3 each
    want = sum(0.5**t * 0.3 for t in range(1, max_turns))
    assert out["episode_return"] == pytest.approx([want] * n)
    assert trainer._metrics["train"]["kernel/correct_pct"] == [100.0]
    # every episode logged for report.py
    import json

    from report import render

    logged = [json.loads(line) for line in (tmp_path / "rollouts-rank0.jsonl").read_text().splitlines()]
    assert len(logged) == n and all(len(ep["completions"]) == len(ep["turns"]) == max_turns for ep in logged)
    assert all(ep['raw_return'] == ep['learning_return'] and len(ep['return_band']) == 2 for ep in logged)
    assert "Turn by turn" in render(logged, "rollouts")


@pytest.mark.skipif(not URL, reason="set KERNEL_ENV_URL to a mock kernel_env server")
def test_run_episodes_against_mock_server():
    eps = run_episodes(lambda convs: [f"```python\n{GOOD}\n```"] * len(convs), URL, [TASK, TASK])
    assert all(t["correct"] for ep in eps for t in ep["turns"])
    assert summarize(eps)["correct_turn1_pct"] == 100


@pytest.mark.skipif(not URL, reason="set KERNEL_ENV_URL to a mock kernel_env server")
def test_pipelined_episodes_against_mock_server():
    import threading

    seen_feedback, lock = [], threading.Lock()

    def chat(conv):
        with lock:
            seen_feedback.append(conv[-1]["content"])
        if len(conv) == 1:  # first turn: cut off mid-module
            return "```python\nimport torch\n# reasoning...", "length", None, 8192
        return f"```python\n{GOOD}\n```", "stop", None, 500

    done = []
    eps = run_episodes_pipelined(chat, URL, [TASK, TASK, "level1/does_not_exist"], on_done=done.append)
    assert len(done) == 3 and "error" in eps[2] and not eps[2]["turns"]
    for ep in eps[:2]:
        assert ep["turns"][0]["truncated"] and ep["turns"][0]["stage"] == "format"
        assert all(t["correct"] and not t["truncated"] for t in ep["turns"][1:])
        assert ep["episode_id"] and len(ep["history"]) == len(ep["turns"])
    assert sum("cut off before the module was complete" in f for f in seen_feedback) == 2


def test_rollout_overlaps_generation_and_evaluation_without_policy_updates(tmp_path, monkeypatch):
    import threading
    import grpo_kernel
    from kernel_env import KernelObservation, KernelState

    entered, release = threading.Event(), threading.Event()
    tok = _tok()

    class Env:
        def __init__(self, **kwargs):
            self.turn = 0

        def sync(self):
            return self

        def reset(self, **kwargs):
            return SimpleNamespace(observation=KernelObservation(problem='P', max_turns=2))

        def step(self, action):
            self.turn += 1
            entered.set()
            assert release.wait(5), 'evaluation blocked the next generation batch'
            obs = KernelObservation(stage='format', feedback='bad source', turn=self.turn,
                                    max_turns=2, done=self.turn==2, correct=False)
            return SimpleNamespace(observation=obs, reward=0.0, done=obs.done)

        def state(self):
            return KernelState(episode_id='test')

        def close(self):
            pass

    class Trainer(FakeTrainer):
        def _generate_single_turn(self, prompts, *args):
            assert len(prompts) <= 2
            if self.calls == 1:
                assert entered.wait(5)
                release.set()  # The first evaluations are still in flight at generation call two.
            return super()._generate_single_turn(prompts, *args)

    monkeypatch.setattr(grpo_kernel, 'KernelEnv', Env)
    trainer = Trainer(tok)
    result = make_rollout_func(KernelArgs(rollout_batch_size=2), str(tmp_path/'s.jsonl'))([TASK]*4, trainer)
    assert len(result['completion_ids']) == 4 and result['episode_return'] == [0.0]*4
    for ids, mask in zip(result['completion_ids'], result['env_mask']):
        assert len(ids) == len(mask) and 0 in mask and 1 in mask
    assert trainer._metrics['train']['rollout/generated_tokens'][0] > 0
    assert trainer.state.global_step == 0


@pytest.mark.skipif(not URL, reason='set KERNEL_ENV_URL to a mock kernel_env server')
def test_paired_episode_seeds_reach_generation_and_logs():
    seeds = []

    def chat(conv, seed):
        seeds.append(seed)
        return GOOD, 'stop', None, 10

    eps = run_episodes_pipelined(chat, URL, [TASK, TASK], seeds=[100, 200])
    assert [ep['sample_seed'] for ep in eps] == [100, 200]
    assert 100 in seeds and 200 in seeds
    with pytest.raises(ValueError):
        run_episodes_pipelined(chat, URL, [TASK], seeds=[])


def test_feedback_ids_qwen35_template():
    # Qwen3.5 is not prefix-stable (earlier assistant turns lose their think block); suffix must still be exact
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    except OSError:
        pytest.skip("Qwen3.5 tokenizer not available offline")
    from grpo_kernel import feedback_ids

    assert tok.decode(feedback_ids(tok, "fb")) == "\n<|im_start|>user\nfb<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@pytest.mark.skipif(not URL, reason="set KERNEL_ENV_URL to a mock kernel_env server")
def test_rollout_unwraps_processor(tmp_path):
    # VLM checkpoints (Qwen3.5) hand the rollout a Processor whose chat template returns nested lists
    tok = _tok()
    trainer = FakeTrainer(tok)
    trainer.processing_class = SimpleNamespace(tokenizer=tok)
    out = make_rollout_func(KernelArgs(env_url=URL), str(tmp_path / "s.jsonl"))([TASK] * 2, trainer)
    assert all(isinstance(t, int) for p in out["prompt_ids"] for t in p)
    assert all(isinstance(t, int) for c in out["completion_ids"] for t in c)
