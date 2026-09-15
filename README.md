# yadro-probes

The fixed diagnostic probe set and activation-extraction code used as the common
reference point across the prescribed-axes line of work.

**Author:** Andrey Lazarev | Independent Researcher

## Why this repository exists

Several results in this line are stated as a comparison against a baseline: cross-model
category structure, coordinate drift between epochs, Procrustes R² between two states of
one model. A comparison needs an object, and that object has to be identifiable by
something other than a folder name.

Until now the probe set lived inside a notebook and the resulting activations lived in a
cloud folder. Neither carried a version. This repository fixes the probe set as an
artifact, records how activations are produced from it, and lists checksums for the stored
activation files so that any later comparison names what it was compared against.

## The probe set

80 prompts in 8 categories of 10: factual, logical, spatial, emotional, abstract, code,
ethical, narrative. Raw completion-style prompts, no chat template applied. Activations are
read from the residual stream at the last token position.

The set was defined in April 2026 and has not been modified since. Any change to it makes
results incomparable with everything measured before the change, so changes go into a new
file rather than an edit.

## Reference result

Step 0 (April 2026), five models: Qwen2.5-3B, Gemma2-2B, OLMo-1B, Falcon-1B, Pythia-1.4B.
Category similarity matrices computed at the middle layer, then correlated pairwise across
models: r = 0.985 to 0.997.

Two relations in the averaged matrix are used downstream as a sanity check that the probe
set reproduces on a new model:

| Pair | Averaged cosine |
|---|---|
| ethical vs spatial | -0.46 |
| logical vs emotional | -0.34 |

Narrative sits apart from the rest; code is close to orthogonal to the others.

These are our own measurements on five models, not an independently replicated finding.
Treat them as a reference for comparison, not as an established property of language models.

## Usage

```
pip install -r requirements.txt
python extract.py --model Qwen/Qwen2.5-0.5B-Instruct --out acts.npy
```

`extract.py` writes an array of shape (80, n_layers+1, hidden) in prompt order, and prints
the category similarity matrix together with the two reference pairs above.

Activation files are not committed. `MANIFEST.md` records what exists, where, and its
SHA-256.

## Open

The step 0 activation files (five models, `yadro_phase2/*.pkl`) were produced before this
repository existed and their checksums are not yet recorded in `MANIFEST.md`. Until they
are, any Procrustes value computed against step 0 names a baseline that cannot be verified
by a third party.

## License

MIT
