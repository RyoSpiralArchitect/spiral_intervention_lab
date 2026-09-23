# Full-Depth Llama Worker Return

Date: 2026-09-14. Controller: `gpt-5.6-luna`.

## Question And Controls

Is the GPT-2 readout-escape bottleneck intrinsic to the task, or specific to the
worker/trajectory/implementation? This is a **worker substitution experiment**,
not an architecture-only intervention: scale, tokenizer, pretraining and
instruction tuning all change together.

The CLI-supplied local Llama checkpoint has 28 layers, width 3072, 24 query
heads, 8 KV heads, vocabulary 128256, and a Llama instruction chat template.
No depth truncation was used. Both workers were measured with greedy decoding,
FP16 on MPS, seed 7, the original constrained-rewrite task, the same raw prompt,
and the existing 18-token/task-stop boundary. Native chat was tested separately
with a frozen template date; it was not silently substituted for the raw prompt.

HF and TransformerLens preflights run in separate processes. They compare
identical input token IDs and teacher-forced HF prefixes, rather than comparing
logits at already-diverged generated prefixes. The runtime codec is checked
against HF encoding with `add_special_tokens=False`. Native chat contains its
own BOS. The model paths are never repository defaults.

## Results

| Worker / Condition | Score | Task Done | Output |
| --- | ---: | --- | --- |
| GPT-2 FP16 / HF raw, no edit | 0.270833 | false | In order to stay on schedule, the budget draft should be sent |
| GPT-2 FP16 / TransformerLens raw, no edit | 0.270833 | false | same as HF |
| GPT-2 FP32 / HF and TransformerLens raw, no edit | 0.270833 | false | same as FP16 |
| Llama / HF raw, no edit | 1.0 | true | Mira will send Omar the budget draft before lunch. |
| Llama / TransformerLens raw, no edit | 1.0 | true | same as HF |
| Llama / HF and TransformerLens native chat, no edit | 1.0 | true | same words as raw, without its leading space |
| Llama / full runtime B0 | 1.0 | true | same as raw preflight |
| Llama / full runtime C1 with Luna | 1.0 | true | byte-identical to B0 |

Llama HF/TransformerLens comparisons pass at all 22 measured prefixes (11 raw,
11 chat): top1 and top20 sets agree throughout, maximum KL is approximately
3.65e-5 and maximum centered-logit RMSE is 0.003931. GPT-2 FP16 also passes the
prespecified distribution check at all 13 prefixes: maximum KL 0.000303,
maximum centered RMSE 0.036438, and identical greedy token sequences. GPT-2's
processed unembedding shifts absolute logits, so the fidelity test reports both
raw differences and distribution-invariant centered errors. It does not mistake
a shared logit offset for distribution corruption.

The additional GPT-2 FP32 control also matches at all 13 teacher-forced prefixes
(maximum centered RMSE 8.09e-5), with the same constraint-violating output as
FP16. The baseline task gap in this comparison is therefore not explained merely
by choosing FP16 for the larger worker.

The C1 episode contains 11 controller `noop` decisions and six diagnostic
requests. No rollout edit was applied, no permission limit was raised, and the
worker produced the answer itself. Control phases were `entity_insertion` six
times and `monitor` five times, **not `readout_escape`**. The existing minimal
suite makes a paired B0 trace available to C1; this is recorded as a condition,
not hidden as if it were the older C1-only experiment.

A diagnostic matched-response matrix did execute: `mlp_out` layer 27, source
term activation, observed-gap-carrier seed, doses 0.04 and 0.16, bound pieces
` send` and ` Send`. Two physical edits plus repeats and controls produce four
observables in six simulations; state restoration passes. All four rows remain
`rank_carrier`, target top20 deltas are zero, and reported term-mass deltas are
approximately zero. A smaller gap sometimes comes from a falling top20 threshold,
not a rising target logit. FP16-sized increments here cannot be compared directly
to the older GPT-2 FP32 micro-deltas or used to claim an actuator.

## Integration Findings

The local HF loader now honors the requested dtype. Llama loading preserves
unprocessed RMSNorm weights and activation coordinates instead of folding them
or upcasting the entire checkpoint for weight processing. GPT-2's existing
processing path remains intact. A tiny full HF/GQA conversion test verifies
logit agreement and retained RMSNorm; the real 28-layer preflight verifies the
actual checkpoint rather than treating the tiny test as model validation.

An initial full-runtime attempt was interrupted after 244 seconds while B0 was
inside its KV observer, before any controller call. Its partial JSONL and a
process sample are preserved, not reported as a scored model failure. The sample
shows repeated device-to-CPU copies; peak footprint was 15.8 GiB in that attempt.

The rerun batches KV cache transfers by layer/site, keeps projection matrices on
CPU for the CPU scan, and selects embedding rows before FP32 conversion. The
source-position scoring/ranking formula and scan coverage are unchanged. Rerun
worker steps finish in roughly 0.6-2.6 seconds, excluding packet construction,
diagnostic replays and provider time; this is an observed timing, not a controlled
performance benchmark. `live_stdout.log` includes a scheduled faulthandler dump
at 120 seconds during a diagnostic: its `Timeout` header is instrumentation, not
a failed episode or a timeout retry.

There was also a **coordinate bug**, distinct from transfer cost:
TransformerLens `attn.hook_k` is pre-RoPE, whereas `attn.hook_rot_k` is post-RoPE.
The observer previously tried rotating its prototype for the former. It now
matches the referenced cache space and uses the rotary buffer's device when a
post-RoPE reference is actually requested. CPU/MPS and GQA tests cover this
boundary. This correction can change diagnostic candidates; older controller
trajectories are not claimed to be bitwise comparable with this rerun.

Validation: `python3 -m pytest SpiralInterventionLab/tests -q` reports **321
passed**, with 17 warnings. The installed TransformerLens also warns about MPS
numerical correctness; the measured HF parity check addresses these short
prefixes only, not every operation or context length.

## Interpretation And Next Step

The original seed-7 task is no longer an intervention-success benchmark for this
worker: its baseline is already at ceiling. The result shows usable full-depth
Llama integration and no observed controller regression in this episode. It does
not establish that an operator improved the answer, that GPT-2's architecture
alone caused collapse, or that the framework generalizes. The new GPT-2 no-edit
control is coherent but constraint-violating, not a universal `the the` repeater.

The Luna debrief correctly notes success without intervention, but still frames
absent apply permission as a possible "failure factor." Treat that as qualitative
interface feedback, not causal evidence or a reason to relax a gate. Six
diagnostics during natural progress may be unnecessary; a progress-aware
diagnostic-demand ablation is a separate future experiment.

Next, keep this case as a no-regression control. Prespecify a small Llama task
difficulty ladder and find conditions with genuine no-edit failures but coherent
generation. Then run paired B0/C1 measurements at that difficulty, keeping
token-binding and source identity fixed within each worker. Do not manufacture
collapse, transplant GPT-2 layer/head/token IDs, or increase apply authority just
to make the new model require intervention.

## Artifacts And Rerun

[Audit and hashes](../results/llama_return_20260914/audit.json),
[C1 JSONL](../results/llama_return_20260914/luna_live_r2/c1.jsonl),
[B0/C1 summary](../results/llama_return_20260914/luna_live_r2/experiment_summary.json),
and [qualitative debrief](../results/llama_return_20260914/luna_live_r2/post_run_debrief.md)
are retained locally, including raw last-token logits and the interrupted attempt.

```sh
python3 -m SpiralInterventionLab.examples.worker_fidelity_preflight \
  --backend hf --worker-model-path "$LLAMA_MODEL_PATH" \
  --output-dir results/llama_preflight/hf
python3 -m SpiralInterventionLab.examples.worker_fidelity_preflight \
  --backend tlens --worker-model-path "$LLAMA_MODEL_PATH" \
  --reference-dir results/llama_preflight/hf \
  --output-dir results/llama_preflight/tlens

PYTHONPATH=. python3 -u results/llama_return_20260914/run_live.py \
  --provider openai --controller-model gpt-5.6-luna \
  --worker-model meta-llama/Llama-3.2-3B-Instruct \
  --worker-model-path "$LLAMA_MODEL_PATH" --worker-hf-offline \
  --worker-device mps --worker-dtype float16 --worker-mps-mode conservative \
  --task constrained_rewrite --seed 7 --no-b1 \
  --controller-prompt-profile compact --controller-packet-view compact \
  --readout-analyzer sae_scaffold --readout-analyzer-rerank-mode apply \
  --activation-surface-profile activation_patch_expanded \
  --max-diagnostic-calls-per-run 12 --diagnostic-result-window 12 \
  --post-run-debrief controller --post-run-debrief-max-output-tokens 1200 \
  --log-dir results/llama_rerun
```

Verify the preflight reports before starting C1. Logit tolerances are an
integration smoke boundary, not an operator-effect significance test. The live
wrapper disables autograd and adds timing prints only; it delegates decisions,
budgets, scoring and logging to the existing CLI.
