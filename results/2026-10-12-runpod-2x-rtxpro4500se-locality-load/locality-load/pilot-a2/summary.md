# Locality versus load — measured cohort

Source: `/root/results/inference-under-pressure/locality-load/pilot-a2`. 1 repetitions; each policy has the same model, host, session transcripts and timing schedule within a repetition.

| Profile | Policy | Prefill vs affinity | Resumed TTFT p95 (ms) | Wall (s) | Queue (ms) | Preemptions | Placement skew | In-flight imbalance | Output tokens median |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | R0 | 4.99× | 290.0 | 34.6 | 0.01 | 0 | 0.00 | 0.133 | 32 |
| reference | R1 | 3.61× | 279.7 | 31.3 | 0.16 | 0 | 0.01 | 0.099 | 32 |
| reference | R1S4 | 1.00× | 51.2 | 26.9 | 0.01 | 0 | 0.00 | 0.218 | 32 |
| reference | R2 | 1.00× | 49.9 | 27.4 | 0.01 | 0 | 0.00 | 0.237 | 32 |
| reference | R3 | 1.00× | 49.1 | 27.4 | 0.01 | 0 | 0.06 | 0.248 | 32 |

| saturation | R0 | 6.16× | 555.1 | 134.7 | 5.60 | 0 | 0.00 | 0.026 | 512 |
| saturation | R1 | 2.42× | 302.8 | 128.0 | 0.01 | 0 | 0.01 | 0.035 | 512 |
| saturation | R1S4 | 1.08× | 64.3 | 125.6 | 0.01 | 0 | 0.01 | 0.049 | 512 |
| saturation | R2 | 1.00× | 64.3 | 125.4 | 0.01 | 0 | 0.00 | 0.035 | 512 |
| saturation | R3 | 1.00× | 63.9 | 125.5 | 0.01 | 0 | 0.00 | 0.035 | 512 |

| hot-skew | R0 | 5.81× | 318.5 | 149.0 | 0.01 | 0 | 0.00 | 0.035 | 512 |
| hot-skew | R1 | 2.11× | 298.2 | 143.6 | 0.17 | 0 | 0.01 | 0.053 | 512 |
| hot-skew | R1S4 | 1.00× | 62.5 | 141.6 | 0.01 | 0 | 0.00 | 0.053 | 512 |
| hot-skew | R2 | 1.00× | 64.1 | 142.0 | 0.01 | 0 | 0.00 | 0.064 | 512 |
| hot-skew | R3 | 1.00× | 65.1 | 142.2 | 0.01 | 0 | 0.00 | 0.079 | 512 |

**Interpretation rule:** Zero qualifying profiles means no locality/load crossover was demonstrated. Check measured queueing, realized output lengths and skew before claiming that the hardware was saturated. These are repeated synthetic traces, not independent production traffic populations.
