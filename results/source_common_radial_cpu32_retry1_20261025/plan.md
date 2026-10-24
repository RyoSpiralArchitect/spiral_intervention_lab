# Authorized Same-Condition Retry

The cooperator requested this new attempt after the competing local run ended.
The incomplete first attempt remains untouched. This is one explicit retry,
not an implicit retry of a failed condition or an outcome-adaptive sweep.

Reuse the previous ten-condition plan without changing model checkpoint, prompt,
natural position-8 prefix, exact SOURCE IDs 11/1418, source positions, layer/site,
alpha 0.04, step cap 0.08, TTL 1, CPU/float32, four threads, seed 0 or the 12-slot
diagnostic budget. Keep the same 900-second, 32-GiB-start and 16-GiB-stop limits.

Record one new set of raw vectors and receipts. Compare exact input/captured
vector identities against both the previous completed CPU32 controls and the
five completed conditions from the interrupted attempt. Recompute component
geometry and raw full-vocabulary readout without extra model forwards.

No controller API, production apply, checkpoint download, adaptive site/strength
search, or interruption of another process is authorized by this protocol.
GPT-2 is a separate comparison with its own tokenizer/checkpoint identity and
explicit prefix provenance. Its model process must run after this child exits,
not concurrently, to avoid recreating the memory-pressure confound.
