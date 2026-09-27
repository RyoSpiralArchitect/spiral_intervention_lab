# Early blueprint offer: fixed-prefix controller choice (2026-09-26)

## Question and method

Does exposing the early `activation_patch_candidate_review` option change the
controller's next diagnostic choice, holding the answer prefix and recorded
predecessor evidence fixed? The shadow replay reconstructs one worker state
from the first two charged entity diagnostics of each live C1 JSONL. It builds
`off` and `soft` packets from that same state, checks that only handoff/offer
fields differ, and invokes the same Luna controller twice per condition in
counterbalanced order (`off,soft`, then `soft,off`). No requested diagnostic is
dispatched, no edit is applied, and generation does not advance after the
chosen prefix. This compares controller choices, not task outcomes.

The executable harness is
`SpiralInterventionLab/examples/replay_candidate_seed_discovery_choice.py`.
Checkpoint paths are CLI inputs. Each report records source JSONL, prompt,
model-config, predecessor-result, and compact-packet hashes. Both replayed
soft packets reproduced the live source's two early review options and its
canonical `operator_diagnostic_replay` frontier at the same prefix.

For GPT-2, with the recorded C1 JSONL and checkpoint available locally:

```sh
python3 -m SpiralInterventionLab.examples.replay_candidate_seed_discovery_choice \
  --profile gpt2_control --source-jsonl "$SOURCE_JSONL" \
  --worker-model-path "$MODEL_PATH" --prefix ' In' \
  --objective entity_insert:mira:source_body:weak_reachable \
  --output "$OUTPUT"
```

Use `--profile llama_l1`, prefix ` Nora`, and objective
`entity_insert:sample:source_body:near_reachable` for the Llama condition.
The receipt hashes identify the exact local source episodes and full reports;
the large raw JSONL and provider traces are intentionally not committed.

## Observed choices

| Worker | Prefix | Early soft offers | Off choice | Soft choice |
| --- | --- | --- | --- | --- |
| GPT-2, seed 7 | ` In` | Mira, Omar | canonical replay 2/2 | canonical replay 2/2 |
| Llama-3.2-3B-Instruct, seed 0 | ` Nora` | sample, Selim | canonical replay 2/2 | canonical replay 2/2 |

In both models, the canonical option was shown at priority 10 and the early
activation-patch reviews at priority 20. All eight controller calls returned a
`noop` command with a request for `operator_diagnostic_replay`; the raw command
and extracted request agree on this diagnostic. An initial local GPT-2 report
incorrectly labeled these as `other_diagnostic` because the canonical frontier
was represented as a string rather than a request object. Its raw requests
were unchanged; that preliminary report is not included in this PR. The
corrected full reports remain local as
`results/early_seed_choice_gpt2_20260926/controller_choices_v2.json` and
`results/early_seed_choice_llama_20260926/controller_choices.json`. The PR
includes compact, source-hashed receipts at
`results/early_seed_choice_gpt2_20260926/receipt.json` and
`results/early_seed_choice_llama_20260926/receipt.json`, without thousands of
lines of provider trace.

This is narrow evidence that simply surfacing the early blueprint review did
not change the next choice in these fixed-prefix states. It does not show that
activation patching is ineffective, that the controller can never choose the
offer, or that altering option priority would improve task score. The
reconstruction reuses recorded diagnostic result payloads, not a byte-for-byte
snapshot of all live worker history. The MPS backend also warns of potential
silent incorrectness. No production permission changed.
