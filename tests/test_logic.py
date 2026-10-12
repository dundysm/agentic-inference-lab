"""CPU-only correctness tests. Does not validate vLLM metrics or GPU runs."""
import argparse
import importlib.util
import json
import tempfile
import os
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
routing = load(ROOT / "scripts" / "routing_analyze.py", "routing_analyze")


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

    def test_worker_id_parsing(self):
        self.assertEqual(wl.get_worker_id({"nvext": {"worker_id": {"decode_worker_id": 7}}}), 7)
        self.assertEqual(wl.get_worker_id({"nvext": {"worker_id": 3}}), 3)
        self.assertIsNone(wl.get_worker_id({"choices": []}))
        self.assertIsNone(wl.get_worker_id({"nvext": {"worker_id": {"decode_worker_id": True}}}))

    def test_routing_flags_do_not_change_prompt_digest(self):
        # Routing metadata travels in headers and nvext, never in the prompt text.
        base = wl.prompt_set_digest(self.args())
        day2 = wl.prompt_set_digest(self.args(route_arm="R2", session_header=True,
                                              request_worker_id=True))
        self.assertEqual(base, day2)

    def test_default_schedule_is_fixed_timing(self):
        sched = wl.timing_schedule(self.args(pause=0.5, stagger_s=0.0, pause_max=0.0, schedule_seed="s"))
        for entry in sched.values():
            self.assertEqual(entry["start_delay_s"], 0.0)
            self.assertEqual(entry["pauses_s"], [0.5, 0.5])

    def test_sensitivity_schedule_is_seeded_and_bounded(self):
        a = self.args(sessions=16, turns=12, pause=0.25, stagger_s=8.0, pause_max=1.75, schedule_seed="s")
        one, two = wl.timing_schedule(a), wl.timing_schedule(a)
        self.assertEqual(one, two)
        for entry in one.values():
            self.assertTrue(0 <= entry["start_delay_s"] <= 8.0)
            self.assertEqual(len(entry["pauses_s"]), 11)
            self.assertTrue(all(0.25 <= p <= 1.75 for p in entry["pauses_s"]))
        self.assertNotEqual(one, wl.timing_schedule(self.args(sessions=16, turns=12, pause=0.25,
                                                              stagger_s=8.0, pause_max=1.75, schedule_seed="t")))
        # Timing never touches the prompts.
        self.assertEqual(wl.prompt_set_digest(a), wl.prompt_set_digest(self.args(sessions=16, turns=12)))

    def test_metric_targets(self):
        out = Path("x")
        day1 = wl.metric_targets(argparse.Namespace(metrics=[], base_url="http://h:8000", out_dir=out), "before")
        self.assertEqual(day1, [("http://h:8000/metrics", out / "metrics_before.prom")])
        day2 = wl.metric_targets(argparse.Namespace(
            metrics=["frontend=http://h:8000/metrics", "worker0=http://h:8081/metrics"],
            base_url="http://h:8000", out_dir=out), "after")
        self.assertEqual(day2, [("http://h:8000/metrics", out / "frontend_metrics_after.prom"),
                                ("http://h:8081/metrics", out / "worker0_metrics_after.prom")])

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

    def test_renamed_routing_analysis_imports(self):
        # These used to import sibling files with chronological names.
        for filename, name in (
            ("routing_evaluate.py", "routing_evaluate_smoke"),
            ("routing_followup_evaluate.py", "routing_followup_evaluate_smoke"),
            ("routing_plots.py", "routing_plots_smoke"),
        ):
            self.assertIsNotNone(load(ROOT / "scripts" / filename, name))

    def test_inflight_imbalance_is_time_weighted(self):
        xo = load(ROOT / "scripts" / "analyze_load_crossover.py", "crossover_amend1")
        wmap = {10: 0, 11: 1}
        req = lambda w, a, b: {"worker_id": w, "start_unix": a, "end_unix": b}
        # Perfectly balanced: one request on each worker at all times.
        self.assertAlmostEqual(xo.inflight_imbalance([req(10, 0, 4), req(11, 0, 4)], wmap), 0.0)
        # Everything on one worker: maximal imbalance.
        self.assertAlmostEqual(xo.inflight_imbalance([req(10, 0, 4), req(10, 1, 3)], wmap), 0.5)
        # Equal request COUNTS per worker but bursty in time: worker 0 alone for 0-2 s,
        # balanced 2-4 s, worker 1 alone 4-6 s -> (0.5*2 + 0*2 + 0.5*2) / 6.
        bursty = [req(10, 0, 4), req(11, 2, 6)]
        self.assertAlmostEqual(xo.inflight_imbalance(bursty, wmap), 2 / 6)
        # Idle gaps are excluded from the denominator.
        gap = [req(10, 0, 1), req(11, 5, 6)]
        self.assertAlmostEqual(xo.inflight_imbalance(gap, wmap), 0.5)

    def test_crossover_gate_and_wall_tolerance(self):
        xo = load(ROOT / "scripts" / "analyze_load_crossover.py", "crossover_amend1_gate")
        def cells(r2_queue=0.0, r2_preempt=0, r2_imb=0.25, cand_p95=80.0, cand_wall=101.0):
            return {"R2": {"queue_mean_ms": r2_queue, "preemptions": r2_preempt, "inflight_imbalance": r2_imb,
                           "ttft_p95_ms": 100.0, "wall_s": 100.0},
                    "R1": {"ttft_p95_ms": cand_p95, "wall_s": cand_wall}}
        # No server-side contention -> never a crossover, however uneven R2's in-flight load looks
        # (affinity is burstier than round robin even when idle; imbalance is descriptive only).
        self.assertFalse(xo.crossover_win(cells(), "R1"))
        self.assertFalse(xo.crossover_win(cells(r2_imb=0.45), "R1"))
        # Queueing on R2 opens the gate; 1% slower wall is within the 2% tolerance.
        self.assertTrue(xo.crossover_win(cells(r2_queue=6.0), "R1"))
        # Any preemption on an R2 worker also opens it.
        self.assertTrue(xo.crossover_win(cells(r2_preempt=1), "R1"))
        # More than 2% slower, or under 10% better p95, fails.
        self.assertFalse(xo.crossover_win(cells(r2_queue=6.0, cand_wall=103.0), "R1"))
        self.assertFalse(xo.crossover_win(cells(r2_queue=6.0, cand_p95=91.0), "R1"))

    def test_ignore_eos_changes_digest_only_when_set(self):
        base = self.args(sessions=16, turns=12, lines=80)
        off = self.args(sessions=16, turns=12, lines=80, ignore_eos=False)
        on = self.args(sessions=16, turns=12, lines=80, ignore_eos=True)
        self.assertEqual(wl.prompt_set_digest(base), wl.prompt_set_digest(off))
        self.assertNotEqual(wl.prompt_set_digest(base), wl.prompt_set_digest(on))

    def test_amendment2_stress_profiles_force_512_tokens(self):
        env = dict(os.environ, DRY_RUN="1")
        try:
            a2 = subprocess.check_output(["bash", "scripts/run_load_crossover.sh"], cwd=ROOT,
                                         env=dict(env, STUDY_PHASE="pilot-a2"), text=True)
            original = subprocess.check_output(["bash", "scripts/run_load_crossover.sh"], cwd=ROOT,
                                               env=dict(env, STUDY_PHASE="pilot"), text=True)
        except (OSError, subprocess.CalledProcessError) as exc:  # Windows without a POSIX bash on PATH
            self.skipTest(f"bash unavailable: {exc}")
        for line in a2.splitlines():
            if "profile=reference" in line:
                self.assertIn("max_output=32 ignore_eos=0", line)
            elif "profile=" in line:
                self.assertIn("max_output=512 ignore_eos=1", line)
        # The already-run original pilot keeps its as-run settings for auditability.
        for line in original.splitlines():
            if "profile=saturation" in line or "profile=hot-skew" in line:
                self.assertIn("max_output=256 ignore_eos=0", line)

    def test_crossover_output_cap_changes_digest_only_when_changed(self):
        base = self.args(sessions=16, turns=12, lines=80)
        same = self.args(sessions=16, turns=12, lines=80, max_output_tokens=32,
                         hot_session_count=8, hot_pause_scale=0.1, cold_pause_scale=3)
        other = self.args(sessions=16, turns=12, lines=80, max_output_tokens=256)
        self.assertEqual(wl.prompt_set_digest(base), wl.prompt_set_digest(same))
        self.assertNotEqual(wl.prompt_set_digest(base), wl.prompt_set_digest(other))

    def test_hot_cold_timing_is_seeded_without_affecting_baseline(self):
        args = self.args(sessions=8, pause=1, pause_max=0, stagger_s=0,
                         schedule_seed="test", hot_session_count=3,
                         hot_pause_scale=0.1, cold_pause_scale=3)
        a = wl.timing_schedule(args)
        self.assertEqual(a, wl.timing_schedule(args))
        self.assertEqual({sid for sid, v in a.items() if v["is_hot"]}, {0, 2, 4})
        self.assertEqual(a[0]["pauses_s"], [0.1, 0.1])
        self.assertEqual(a[1]["pauses_s"], [3.0, 3.0])
        normal = wl.timing_schedule(self.args(sessions=8, pause=1, pause_max=0,
                                             stagger_s=0, schedule_seed="test"))
        self.assertNotIn("is_hot", normal[0])
        self.assertEqual(normal[0]["pauses_s"], [1, 1])

    def test_crossover_matrix_is_shuffled_and_reproducible(self):
        import os
        cmd = ["bash", "scripts/run_load_crossover.sh"]
        env = dict(os.environ, DRY_RUN="1", STUDY_PHASE="pilot")
        a = subprocess.check_output(cmd, cwd=ROOT, env=env, text=True)
        b = subprocess.check_output(cmd, cwd=ROOT, env=env, text=True)
        self.assertEqual(a, b)
        rows = [x for x in a.splitlines() if x.startswith("rep=")]
        self.assertEqual(len(rows), 15)
        self.assertEqual({(x.split()[1], x.split()[2]) for x in rows},
                         {(f"profile={p}", f"arm={arm}")
                          for p in ("reference", "saturation", "hot-skew")
                          for arm in ("R0", "R1", "R1S4", "R2", "R3")})

    def test_routing_missing_prefill_counter_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "manifest.json").write_text(json.dumps(
                {"workers": 2, "arm": "R0", "rep": 1, "completed": True}))
            (d / "worker_map.json").write_text('{"42": 0}')
            (d / "requests.jsonl").write_text(json.dumps({
                "session": 0, "turn": 0, "worker_id": 42, "success": True,
                "ttft_ms": 10, "user_ttft_ms": 10, "start_unix": 1,
                "end_unix": 2, "prompt_set_sha256": "test",
            }) + "\n")
            with self.assertRaisesRegex(ValueError, "required prefill counter"):
                routing.summarize(d, 500)


if __name__ == "__main__":
    unittest.main()
