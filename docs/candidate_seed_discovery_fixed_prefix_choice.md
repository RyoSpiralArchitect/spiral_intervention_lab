# Early blueprint offer: fixed-prefix controller choice (2026-09-26)

## Question and method

Does exposing the early `activation_patch_candidate_review` option change the
controller's next diagnostic choice, holding the answer prefix and recorded
predecessor evidence fixed? The shadow replay reconstructs one worker state
from the first two charged entity diagnostics of each live C1 JSONL. It builds
`off` and `soft` packets from that same state, checks that only handoff/offer
fields differ, then removes the condition-dependent `source_packet_sha256`
from the precompacted controller input. The raw hashes remain in the audit
report. It invokes the same Luna controller twice per condition in
counterbalanced order (`off,soft`, then `soft,off`). No requested diagnostic is
dispatched, no edit is applied, and generation does not advance after the
chosen prefix. This compares controller choices, not task outcomes.

The executable harness is
`SpiralInterventionLab/examples/replay_candidate_seed_discovery_choice.py`.
Checkpoint paths are CLI inputs. Each report records source JSONL, prompt,
model-config, predecessor-result, raw compact-packet, and actual controller-input
hashes. Both reconstructed soft packets reproduced the live source's two early
review options and its canonical `operator_diagnostic_replay` frontier at the
same prefix. The harness also checks that existing diagnostic options retain
their order and priority.

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

| Worker | Prefix | Input status | Early soft offers | Off choice | Soft choice |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-3B-Instruct, seed 0 | ` Nora` | hash-blind | sample, Selim | canonical replay 2/2 | canonical replay 2/2 |
| GPT-2, seed 7 | ` In` | historical, hash-confounded | Mira, Omar | canonical replay 2/2 | canonical replay 2/2 |

In both models, the canonical option was shown at priority 10 and the early
activation-patch reviews at priority 20. The four hash-blind Llama calls all
returned `noop` with `operator_diagnostic_replay`; the raw command and extracted
request agree. The older off/soft comparison also returned canonical replay
in both models, but its compact packet included a different opaque source hash
per condition. It cannot independently establish the effect of showing the
offer. GPT-2's local checkpoint subsequently lost its `model.safetensors`, so
the hash-blind comparison could not be rerun for GPT-2. No checkpoint file was
changed by this work.

The hash-blind Llama raw report remains local at
`results/early_seed_choice_llama_20260926/hash_blind_controller_choices.json`;
its committed receipt is `results/early_seed_choice_llama_20260926/receipt.json`.
The earlier GPT-2 and Llama runs are retained as historical receipts named
`receipt_pre_hash_normalization.json` in their respective result directories.
An additional initial GPT-2 local report mislabeled canonical replay as
`other_diagnostic` because the frontier was represented as a string; its raw
requests were unchanged and it is not included in this PR. Full JSONL and
provider traces remain local, with SHA-256 links in the receipts.

The controlled Llama result is narrow evidence that surfacing the early
blueprint review did not change the next choice in this fixed-prefix state.
The GPT-2 observation is suggestive but remains hash-confounded. Neither shows that
activation patching is ineffective, that the controller can never choose the
offer, or that altering option priority would improve task score. The
reconstruction reuses recorded diagnostic result payloads, not a byte-for-byte
snapshot of all live worker history. The MPS backend also warns of potential
silent incorrectness. No production permission changed.
