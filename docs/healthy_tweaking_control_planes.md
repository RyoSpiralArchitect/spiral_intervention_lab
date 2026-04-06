# Healthy Tweaking Control Planes

This project keeps the worker as the primary actor. The controller and runtime may shape the worker's search, but they should not replace the worker's task-solving role.

## Mainline

Mainline controls are the default research path. They should preserve the "worker solves, controller steers" framing.

| Plane | Examples | Why it stays mainline |
| --- | --- | --- |
| Internal intervention DSL | `resid_add`, `kv_mix`, `rank1_patch`, bounded budgets | Core research question: can runtime internal control help without direct answer injection? |
| Structured reflection | `controller_memory`, effect summaries, hypothesis tracking | Improves intervention search without giving the controller a hidden task-solving channel. |
| Task-grounded feedback | `partial_score`, `required_term_recall`, `forbidden_term_clean`, `budget_ok` | Makes "progress > stability" explicit without changing who solves the task. |

## Auxiliary

Auxiliary controls are healthy supports. They can make the worker easier to steer, but should remain soft, local, and auditable.

| Plane | Current modes | Intended use |
| --- | --- | --- |
| Task-agnostic decoder rescue | `loop_aware` | Break short repetition loops and local attractors without task-specific token forcing. |
| Search widening | `loop_aware_prune` | Demote the current overconfident top token once the worker is stalled, so a nearby alternative can surface. |
| Soft explicit-constraint bias | `loop_aware_constraint` | Gently bias toward tokens tied to explicit missing constraints, but never force a full answer string. |
| Soft entity recall prior | `loop_aware_entity_recall` | Continue or start explicit missing entity token sequences from task feedback, while keeping the effect bounded and auditable. |
| Soft entity logit bias | `logit_bias_entity_soft` | Nudge logits toward a small coverage-ranked set of explicit missing entities, stronger for untouched terms and weaker once a term starts to emerge. |
| Semantic progress critic | `MiniLM` sentence critic | Add a coverage-weighted semantic-progress signal so meaning-level drift only counts once explicit task coverage begins to move. |

Auxiliary controls should only:
- operate with bounded, inspectable strength
- use task feedback that is already explicit in the packet
- avoid hidden channels for answer content

## Forbidden

These controls are intentionally out of scope for the main research line.

| Pattern | Why it is forbidden |
| --- | --- |
| Hard token forcing of task answers | Turns the controller into the real generator. |
| Direct answer injection through controller memory or hidden metadata | Violates the worker-as-actor premise. |
| Free-form CoT reinjection carrying task content | Creates an unbounded covert channel. |
| Oversized edits used as bandwidth rather than steering | Blurs intervention into communication. |

## Soft-To-Hard Progression

The intended progression is:

1. Diagnose with task-grounded feedback.
2. Search with bounded internal edits and soft auxiliary controls.
3. Commit only after a direction repeatedly helps.

Strong commits such as weight editing belong at the end of that progression, not at the start.
