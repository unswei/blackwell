# Evaluate models through their decisions

Prediction RMSE measures forecast error. It does not tell you whether that
error changes a chosen action, removes a useful candidate, or predicts task
completion incorrectly. Evaluate those outcomes alongside prediction accuracy
using the same initial state, task and candidate bank for each model.

Blackwell supplies two small numerical helpers in `blackwell.metrics`:
`selection_regret` and `binary_decision_counts`. They consume arrays and return
immutable `NamedTuple` PyTrees. Your experiment owns rollouts, objectives,
constraints and the selection rule. Timing extraction and paired reporting
remain example code, where their task assumptions can be inspected.

These additions are currently unreleased. From a development checkout, run:

```console
uv run python examples/decision_relevant_model_evaluation.py
uv run python examples/decision_relevant_model_evaluation.py --json decisions.json
```

The example prints prediction errors, selection regret, constraint/completion
counts and paired executions. The JSON record preserves per-root masks, model
parameters, candidate IDs, completion times, and requested/executed commands.

## Define one evaluation root

A **root** consists of an initial state, task and shared candidate bank. For
one root, store candidate quantities in arrays of shape `(K,)`, where `K > 0`
is a fixed capacity. A candidate may be a single action or a complete input
sequence. Keep these Boolean masks separate:

| Mask | Meaning |
| --- | --- |
| `candidate_mask` | The slot contains a candidate |
| Model acceptance | The model predicts the declared criterion is met |
| `cost_observed` | The reference cost is available |
| `constraint_observed` | The reference constraint label is available |
| Completion availability | The reference completion label is available |

Padding is excluded by `candidate_mask`. A model-rejected candidate still
exists and must remain in the bank when evaluating model errors. A collision
can establish constraint failure before the final cost or completion time is
available. Do not use one generic “rollout complete” mask for every metric.

Use matching objectives, units, criteria and horizons for predicted/reference
labels. Record thresholds, candidate/root IDs, dwell rules and timing tolerance
with the experiment. The library does not choose them.

## Measure regret for the actual selection

Pass the index selected by your actual rule. It may prioritise arrival,
constraints or another criterion; the metric never replaces it with predicted
cost minimisation:

```python
import jax
import jax.numpy as jnp

from blackwell import metrics

result = jax.jit(metrics.selection_regret)(
    reference_costs=jnp.array([2.0, 1.0, 0.0, 0.0]),
    reference_accepted=jnp.array([True, True, False, False]),
    selected_index=jnp.array(0),
    candidate_mask=jnp.array([True, True, True, False]),
    cost_observed=jnp.array([True, True, False, False]),
    constraint_observed=jnp.array([True, True, True, False]),
)
assert result.regret_defined
assert result.regret == 1.0
assert result.best_reference_index == 1
```

The reference set contains occupied candidates with an observed finite cost
and an observed acceptable constraint label. Regret is the selected reference
cost minus the best cost in that set. This is a **finite-bank reference**, not
a global optimum. Ties use the first eligible candidate in bank order.

Read the flags together:

- `reference_available` says the feasible reference set is nonempty.
- `selection_available` says the supplied index is in range and occupied.
- `selected_cost_available` says its reference cost is observed and finite.
- `selected_constraint_observed` distinguishes missing truth from rejection.
- `selected_feasible` says its observed constraint label accepts.
- `regret_defined` says the selected candidate belongs to the reference set.

No selection (`-1`), any other negative index, an out-of-range index, or a
padded index is unavailable. Such indices are clamped before internal access
and never wrap into another candidate. An infeasible selection is reported
separately from regret. Undefined regret and best cost use **zero placeholders**;
an unavailable best index uses `-1`. Always retain the accompanying flags.

To measure filtering loss, compare the best reference cost from the complete
bank with the best after intersecting `candidate_mask` with model acceptance.
Keep the original mask for selection regret. The example reports the difference
only when both sets are nonempty and records surviving-set availability.

## Count false acceptance and rejection

```python
counts = metrics.binary_decision_counts(
    predicted_accepted=jnp.array([True, False, True, False]),
    reference_accepted=jnp.array([True, True, False, False]),
    candidate_mask=jnp.array([True, True, True, False]),
    constraint_observed=jnp.array([True, True, True, False]),
)
```

There is one correct acceptance, one false acceptance and one false rejection
in this example. The fourth slot is padding.

| | Reference accepts | Reference rejects |
| --- | --- | --- |
| Model accepts | `correct_acceptance` | `false_acceptance` |
| Model rejects | `false_rejection` | `correct_rejection` |

Only occupied slots with observed reference labels contribute. Counts retain
the compared, unobserved and excluded totals, plus rate denominators:

- `false_acceptance_rate` divides false accepts by **observed model accepts**.
- `false_rejection_rate` divides false rejects by **observed reference accepts**.

A zero denominator produces a zero placeholder and a false corresponding
`*_rate_defined` flag. `excluded_count` includes both padding and occupied
slots with missing labels; `unobserved_count` counts only the latter.

For a reserve margin, compare predicted reserve compliance with reference
compliance under **that same margin** first. Also record physical feasibility.
Comparing reserve-based acceptance against physical labels answers a different
question: rejections can come from deliberate reserve conservatism. The example
reports both matched comparisons and a separately labelled reserve-versus-
physical comparison so they can be distinguished.

The same helper can compare completion-by-horizon labels. Pass the completion
availability mask as `constraint_observed`; both sides must refer to the same
completion criterion and horizon.

## Audit completion windows and event times separately

The example defines completion as three consecutive sampled states within
0.10 m of the goal and at speed at most 0.04 m/s. With a 0.25 s sample interval,
the first and last samples span 0.50 s. These are sampled conditions; the
example makes no assertion about continuous motion between them.

The first satisfying window gives an event time. A predicted event is a claim
whose **actual claimed dwell window** must also be checked. An earlier actual
event does not show that the task conditions still hold at the predicted time.
The example keeps these quantities separate:

| Output | Interpretation |
| --- | --- |
| Claim, observable claim, confirmed claim | Was completion predicted, was its window observed, and did that window satisfy the criterion? |
| Timing error and its validity | `actual_time - predicted_time` when both events exist |
| Early / delayed / on-time | Event timing relative to the declared ±0.25 s tolerance |
| Unconfirmed through horizon | A claim with no actual event through a fully observed, nonfailed 6 s branch |
| Unobservable claim | Its claimed window ends after observation ends |
| Right-censored claim | No event observed before a nonfailed recording ends early |
| Physical failure | A terminating sampled collision, reported regardless of a completion claim |

These diagnostics can overlap. A right-censored claim can also be unobservable.
Unobservable claims caused by physical termination remain labelled as failures,
not censoring. An early event can coexist with a missed claimed window.
Unit tests cover exact, early, delayed, unobservable, unconfirmed and failed
outcomes, including leaving the goal after an earlier completion event.

## Pair the executions at the root level

The supplied comparison selects one **complete open-loop sequence** per model.
Execution success requires an observed completion event and no physical failure
through the complete horizon. An event in a truncated recording alone does
not establish that outcome; the raw event is retained separately.
Each reference execution begins from exactly the same complete Euclidean state.
The dynamics are deterministic and have no hidden state or random disturbance;
same-state replay and replay of each selected sequence are tested.

Paired records retain each method's observed success, physical failure, timeout,
no-selection and censoring outcomes. Among pairs with observed outcomes they
count both-success, A-only, B-only and neither-success. A censored pair is
reported separately as unobserved and is not counted as a known failure pair.
Arrival-time differences use `B - A` only among both-success pairs, retaining
that denominator and every failure count.

Executed command arrays cover the observed prefix and use zero placeholders
elsewhere; `executed_control_observed` identifies that prefix. There are no
supervisor overrides in this example.

The optional restricted-time summary assigns the 6 s timeout to known
non-arrivals, including physical failures. It remains separate from observed
arrival times. Censored executions and unspecified no-selection fallbacks
have undefined restricted times, with validity masks in the paired differences.

For stochastic references, declare whether costs represent single realisations
or estimated expectations. Pair disturbance keys across methods where meaningful
and use replicated rollouts to compare expected costs. The minimum of noisy
individual rollouts is not an expected-cost optimum. For an external simulator,
restore solver state, policy/controller memory and physical state, and verify
replay in the application. Log requested and executed commands if a supervisor
can intervene.

An open-loop bank reference does not establish the best first action under a
subsequent replanning controller. State explicitly whether a comparison changes
one action, every replan or the entire sequence.

## What the planar example establishes

The state is `[px, py, vx, vy]` in metres and metres per second. Both axes use
`v_next = 0.65 * v + 0.20 * u` and `p_next = p + 0.25 * v_next` in the reference.
Candidate IDs 0–3 hold forward inputs 0.6, 0.9, 1.2 and 1.5 for four steps,
then apply neutral through step 24; ID 4 is padding. Every root begins at zero
position and velocity. Obstacles are circles checked at sampled positions.

The selector minimises terminal squared goal distance plus
`0.002 * sum(controls**2)` among model-predicted 0.03 m reserve accepts. The
input penalty coefficient converts squared command units into squared metres;
the cost is a declared experimental objective. Reference cost is unavailable
after physical termination or incomplete recording. Physical failure is known
as soon as a collision is observed. Reserve rejection is known as soon as an
observed sample violates the reserve, even if the recording later ends early.

Model A (`gain_error`) changes the input gain to 0.21. Model B (`lateral_drift`)
keeps the reference gain but adds 0.01 m to predicted lateral position per step.
No parameters are fitted. The two ranking cases have distant obstacles:

| Root | Model | Position RMSE (m) | Selected ID | Reference regret (m²) | Observed arrival |
| --- | --- | ---: | ---: | ---: | --- |
| Goal x=0.62 m | A: gain error | 0.02825 | 1 | 0.00183 | None by 6 s |
| Goal x=0.62 m | B: lateral drift | 0.14289 | 2 | 0 | 3.25 s |
| Goal x=0.70 m | A: gain error | 0.02825 | 2 | 0 | 3.25 s |
| Goal x=0.70 m | B: lateral drift | 0.14289 | 2 | 0 | 3.25 s |

Near the ranking boundary, the smaller longitudinal error changes the choice.
The larger lateral error contributes the same terminal cost to every candidate
in these unobstructed cases, leaving the cost ordering unchanged. In the second
root both errors leave the selected sequence unchanged. These are constructed
counterexamples to using RMSE alone to infer decision quality.

Two further roots put a circle on the path to exercise clearance errors,
filtering loss and physical failure. The final root ends recording at 2.5 s to
exercise missing truth. Across the five paired roots, one pair has two observed
successes, one has only B succeed, two have neither succeed, and one is censored.
B has one selected physical failure; A has none. The only both-success arrival
difference is 0 s, with denominator one. These designed cases do not establish
a general model ranking or deployment success rate.

In the complete candidate diagnostics, A makes eight completion claims: four
have observable windows, two are confirmed there, two have no event through a
fully observed nonfailed horizon, and four claims are unobservable. Two claims
are right-censored; the other unobservable windows end after physical failure.
B makes no completion claims despite two observed reference arrivals. Counts
retain their masks in JSON rather than converting missing labels to failures.

## Batch and report without hiding missing data

One call evaluates one root. Stack root arrays with fixed capacity and use
`jax.vmap`; wrap it in `jax.jit` explicitly. Shape and dtype checks on the
public metrics run eagerly and during tracing. Discrete labels and selection
indices are evaluation outcomes, not differentiable control objectives.

Sum confusion counts and denominators before deriving pooled rates. Report
regret only with its defined count and selected feasibility/observation flags.
Aggregate uncertainty at the independent root or episode level; candidates
and control ticks within a root are not additional independent trials. The
example reports deterministic cases without confidence intervals. Bootstrap
intervals for a larger study belong in host-side experiment code with explicit
seeds and episode grouping.

The metrics also accept summaries from probabilistic models. They do not
estimate covariance or establish probabilistic safety. See the
[metrics reference](../reference/metrics.md) for exact fields and
[simulation and evaluation](simulation-and-evaluation.md) for RMSE and NEES.
