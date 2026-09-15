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
