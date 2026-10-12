# Locality versus load — measured cohort

Source: `/root/results/inference-under-pressure/locality-load/pilot`. 1 repetitions; each policy has the same model, host, session transcripts and timing schedule within a repetition.

| Profile | Policy | Prefill vs affinity | Resumed TTFT p95 (ms) | Wall (s) | Queue (ms) | Preemptions | Placement skew | In-flight imbalance | Output tokens median |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | R0 | 4.30× | 291.2 | 33.6 | 0.01 | 0 | 0.00 | 0.129 | 32 |
| reference | R1 | 3.20× | 277.8 | 30.0 | 0.01 | 0 | 0.01 | 0.083 | 32 |
| reference | R1S4 | 1.00× | 49.1 | 26.9 | 0.01 | 0 | 0.00 | 0.232 | 32 |
| reference | R2 | 1.00× | 49.3 | 27.6 | 0.01 | 0 | 0.00 | 0.257 | 32 |
| reference | R3 | 1.00× | 49.9 | 27.5 | 0.89 | 0 | 0.00 | 0.219 | 32 |

| saturation | R0 | 4.93× | 397.8 | 41.9 | 1.77 | 0 | 0.00 | 0.119 | 62 |
| saturation | R1 | 2.91× | 290.9 | 38.3 | 0.01 | 0 | 0.00 | 0.091 | 60 |
| saturation | R1S4 | 1.44× | 232.3 | 36.9 | 0.01 | 0 | 0.01 | 0.146 | 61 |
| saturation | R2 | 1.00× | 53.7 | 35.5 | 0.01 | 0 | 0.00 | 0.123 | 62 |
| saturation | R3 | 1.00× | 52.8 | 35.9 | 0.01 | 0 | 0.00 | 0.112 | 61 |

| hot-skew | R0 | 3.66× | 302.9 | 59.5 | 0.01 | 0 | 0.00 | 0.182 | 62 |
| hot-skew | R1 | 1.96× | 279.7 | 54.9 | 0.16 | 0 | 0.03 | 0.179 | 62 |
| hot-skew | R1S4 | 1.00× | 55.2 | 54.6 | 0.01 | 0 | 0.00 | 0.229 | 61 |
| hot-skew | R2 | 1.00× | 57.4 | 54.6 | 0.01 | 0 | 0.00 | 0.194 | 62 |
| hot-skew | R3 | 1.00× | 56.4 | 54.3 | 0.01 | 0 | 0.00 | 0.241 | 62 |

**Interpretation rule:** Zero qualifying profiles means no locality/load crossover was demonstrated. Check measured queueing, realized output lengths and skew before claiming that the hardware was saturated. These are repeated synthetic traces, not independent production traffic populations.
