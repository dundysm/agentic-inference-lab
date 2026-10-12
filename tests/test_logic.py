"""CPU-only correctness tests. Does not validate vLLM metrics or GPU runs."""
import argparse
import importlib.util
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wl = load(ROOT / "workloads" / "multi_turn.py", "multi_turn")
ana = load(ROOT / "scripts" / "analyze.py", "analyze")


class HarnessTests(unittest.TestCase):
    def args(self, **updates):
        data = dict(model="Qwen/Qwen3-4B-Instruct-2507", sessions=2, turns=3,
                    lines=5, session_offset=0, system_tag="")
        data.update(updates)
        return argparse.Namespace(**data)

    def test_warmup_disjoint_from_measured_document(self):
        a = wl.messages_for_session(0, 80)
        b = wl.messages_for_session(9000, 12, "[warmup-not-measured]")
        self.assertNotEqual(a[0], b[0])
        self.assertNotIn("Session 0000 private record", b[1]["content"])

    def test_hash_includes_all_turns_and_templates(self):
        self.assertNotEqual(wl.prompt_set_digest(self.args(turns=2)),
                            wl.prompt_set_digest(self.args(turns=3)))
        self.assertNotEqual(wl.prompt_set_digest(self.args(session_offset=0)),
                            wl.prompt_set_digest(self.args(session_offset=9000)))

    def test_cache_none_stays_none(self):
        self.assertIsNone(wl.get_cached_tokens({"prompt_tokens_details": None}))
        self.assertEqual(wl.get_cached_tokens({"prompt_tokens_details": {"cached_tokens": 0}}), 0)
        self.assertIsNone(ana.frac({"cached_tokens": None, "prompt_tokens": 100}))
        self.assertEqual(ana.frac({"cached_tokens": 0, "prompt_tokens": 100}), 0)

    def test_dry_run_interleaves_budget_pairs_deterministically(self):
        cmd = ["bash", "scripts/run_baseline.sh"]
        env = dict(__import__("os").environ, DRY_RUN="1", REPS="2",
                   KV_BUDGETS="3221225472 6442450944")
        a = subprocess.check_output(cmd, cwd=ROOT, env=env, text=True)
        b = subprocess.check_output(cmd, cwd=ROOT, env=env, text=True)
        self.assertEqual(a, b)
        for line in a.splitlines():
            self.assertEqual(len(line.split()) - 1, 8)
            self.assertEqual(set(line.split()[1:]), {
                f"{budget}:{arm}" for budget in (3221225472, 6442450944) for arm in "ABCD"
            })

    def test_paired_delta_sign(self):
        runs = []
        for arm, ttft, tokens in [("A", 500, 2000), ("B", 200, 1000),
                                  ("D", 800, 4000), ("C", 600, 2500)]:
            runs.append({"arm": arm, "complete": True, "budget_bytes": 100,
                         "repetition": 1, "resumed_ttft_median_ms": ttft,
                         "user_ttft_median_ms": ttft, "prefill_computed_tokens_total": tokens,
                         "goodput_rps_at_slo": 1})
        out = ana.paired_comparisons(runs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["ttft_median_delta_ms"], -300)
        self.assertEqual(out[1]["computed_prefill_tokens_delta"], -1500)


if __name__ == "__main__":
    unittest.main()
