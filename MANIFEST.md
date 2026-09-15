# Activation files

Activation files are large and are not committed. This file records what exists, where it
is stored, and its checksum, so that a result computed against one of them can name what it
was computed against.

Add an entry with:

```
sha256sum <file>
```

## Step 0, April 2026

Five models, 80 probes, hidden states at the last token position, float32.
Stored in Google Drive, folder `yadro_phase2`.

| File | Model | SHA-256 | Recorded |
|---|---|---|---|
| qwen_results.pkl | Qwen2.5-3B | not yet recorded | |
| gemma_results.pkl | Gemma2-2B | not yet recorded | |
| olmo_results.pkl | OLMo-1B | not yet recorded | |
| falcon_results.pkl | Falcon-1B | not yet recorded | |
| pythia_results.pkl | Pythia-1.4B | not yet recorded | |

Notebook that produced them: `llm_geometry_step0_v4.ipynb`, not yet committed here.

## Measured extraction cost

Recorded so that later run budgets are estimated from a measurement rather than a guess.

| Model | Host | Threads | Batch | Time for 80 probes | Per probe |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | WSL2, 4 cores, CPU only | 4 | 8 | 196 s | 2.45 s |

The host was running an unrelated workload at the same time, so this is an upper bound
under contention rather than a clean figure. Two things that do not depend on contention
were measured separately on the same host: dropping the vocabulary projection head cuts
cost to about 0.73 of the full forward pass, and batching by 8 cuts it by a further 0.37.
