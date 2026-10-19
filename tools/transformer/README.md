# Tiny GPT: authoring pipeline

Everything needed to (re)create the model that powers the **Transformer explorer**. This is
author-time tooling: users of the app never run any of it (the trained weights are committed in
`packages/deep/models/tiny-gpt`).

```
corpus.py            synthetic mini-stories with a real grammar (agreement, pronouns, copying)
bpe.py               byte-level BPE tokenizer, written from scratch (GPT-2 style)
gpt.py               GPT-2-style transformer in NumPy: forward AND hand-written backward pass
check_gradients.py   finite-difference check of that backward pass
train.py             CPU training loop (Adam, cosine schedule, grad clipping)
export.py            writes weights.bin + model.json + tokenizer.json + parity fixtures
```

## Reproduce

Requires Python 3.10+ and NumPy only (no PyTorch, no GPU).

```bash
python tools/transformer/check_gradients.py        # backward pass vs finite differences
python tools/transformer/train.py --steps 2500     # ~20-30 min on one CPU core
python tools/transformer/export.py                 # publish into packages/deep
npm test -w @sim/deep                            # the TypeScript engine must match to 1e-8
```

## Design choices

- **Why a tiny model of our own?** It runs in the browser with no download, it can be opened up
  completely, and its tensors can be verified against an independent implementation. It uses the
  GPT-2 architecture unchanged: learned token + position embeddings, pre-LayerNorm blocks, causal
  multi-head attention, GELU MLP (4× width), final LayerNorm, output tied to the token embedding.
- **Why a synthetic corpus?** No licence or download, and the structure is known: names carry a
  gender (`Anna … She`, `Tom … He`), two names become `They`, an object becomes `it`, a plural
  subject takes a bare verb, and a name from earlier in the story can be copied. Those are exactly
  the behaviours the attention maps let you see. Regenerate with another seed or extend the grammar
  in `corpus.py`.
- **Tokenizer.** Byte-level BPE with 255 learned merges (vocabulary 512, including
  `<|endoftext|>`). Frequent words become one token; unseen words break into pieces; any text
  (emoji included) can be encoded because the 256 bytes are always in the vocabulary.
- **Gradients.** The backward pass is hand-written; `check_gradients.py` compares it with finite
  differences for every parameter tensor (worst relative error about 2e-6 in float64).
- **Speed note.** These matrices are tiny, so BLAS multi-threading hurts (3× slower on 16 cores).
  `train.py` pins BLAS to one thread.

## Outputs and parity

`export.py` stores the float32 weights, then runs the **float64** NumPy forward pass on those exact
weights and saves every intermediate tensor for three prompts
(`packages/deep/fixtures/tiny-gpt-parity.json`). The TypeScript transformer
(`packages/deep/src/transformer`) must reproduce embeddings, per-head Q/K/V, attention
probabilities, residual streams, MLP activations and logits to 1e-8, and the TypeScript tokenizer
must reproduce Python's encodings (including lone bytes of multi-byte characters).
