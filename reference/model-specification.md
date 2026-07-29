# Model specification

Exploration chooses a model family. Specification commits to its components,
functional forms, and hyperparameters before implementation or uncertainty
analysis.

The hard part is not the number of variables. It is the interaction among
hyperparameters: every component competes with the others for overlapping
signal, so changing one weight changes what the remaining weights mean. Reduce
that coupled problem by classifying each knob into a tuning tier and applying
only the method appropriate to that tier.

## Specify greedily

Build additively, largest source of variation first:

1. Fit the dominant component or component family.
2. Classify its knobs by observing what changes across an order-of-magnitude
   scan.
3. Fix the settled knobs.
4. Add the next justified component against that backdrop.
5. Stop when a new component does not earn the interaction cost it introduces.

Use nested variance explained to rank additions and `format_report` energy share
as a quick composition summary; see [diagnostics.md](diagnostics.md). Component
count alone is not the cost. Twenty components with two interacting,
holdout-tunable knobs can be easier to specify than five components whose
weights all trade signal back and forth.

## The three tiers

| Tier | Diagnostic | Selection method |
|---|---|---|
| **1: magnitude** | Nearby log-scale values produce equivalent useful fits | Choose a representative value and freeze it |
| **2: reconstruction** | The knob materially changes reconstruction quality | Compare candidates on held-out reconstruction |
| **3: structure** | The target is a feature of the component itself | Fit all data and apply structural acceptance checks |

Classify by the decision the knob controls, not its name or penalty type. A
smoothing weight may be Tier 1 in one decomposition and Tier 3 in another. Give
structural intent precedence: if the scientific target is event count, timing,
amplitude, monotonicity, or another component property, use Tier 3 even when the
knob also changes reconstruction.

## Calibrate the scan before classifying

The builders scale different penalty classes differently to make their exposed
weights **approximately commensurable**. Local L1 analysis penalties such as
`sparse`, `pwc_trend`, and `pwl_trend` are divided by their analyzed length;
global L2² analysis penalties retain their aggregate-roughness scaling; and
synthesis penalties such as `multiperiodic` and `exog_spline` are scaled in
coefficient space. These choices reduce dependence on record length and make it
less likely that different loss classes require search ranges separated by
three or more orders of magnitude.

Exact scales still depend on signal magnitude, noise, operators, and basis
normalization. Start on comparable log-scale ranges and expand only when
needed. Diagnose direction rather than memorizing an anchor: a flattened
component usually means the weight is too high; noise chasing, excess knots, or
too many spikes means it is too low. Calibration establishes the scan range;
the fitted effect determines the tier.

## Tier 1: set by magnitude

Tier-1 knobs are insensitive over a useful range. Choose a representative value
that produces the intended behavior, confirm that nearby values are equivalent
for the task, then freeze it. Do not spend holdout budget finding a spurious
optimum on a flat response.

## Tier 2: tune the reconstruction

Use Tier 2 when reconstruction quality is the decision and candidate values
change it enough to score reliably. `holdout_select` masks known entries,
refits each candidate, and scores their imputed values with RMSE or MAE.

The current helper uses one contiguous central block, reducing the leakage that
random point holdout creates between nearby correlated samples. It is not
universal: seasonal data may need explicit blocks at representative phases, and
one block can give a noisy ranking. Treat small score differences as a flat
result, not evidence for a precise optimum.

Do not holdout-tune a knob on a minor component. The score measures the whole
reconstructed signal, so a scientifically important but low-amplitude component
may barely move it; the apparent optimum will be noise. Freeze an insensitive
knob as Tier 1 or tune a structurally meaningful one as Tier 3.

## Tier 3: tune the component

Tier-3 knobs control a structurally meaningful feature. Reconstruction fit can
support the decision, but it does not define whether the feature is correct.
Examples include:

- the number, location, and amplitude of level or slope changes;
- whether an accumulation/removal component shows coherent buildup and resets;
- whether a monotone or bounded curve follows the expected physical behavior.

Inspect per-role plots, component-versus-driver relationships, and stability
snapshots. Check persistence across nearby weights and data windows, plus
domain-coherent sign, timing, and magnitude. Reconstruction score cannot replace
these checks.

Tier-3 scans normally fit **all observed data**, not a holdout subset. The goal
is to select a structurally credible component, and withholding observations
weakens the evidence for subtle events while still scoring the wrong target.
This is a deliberate distinction from Tier 2, not a missing validation step.

Automate Tier 3 with a component-specific acceptance rule. For a
piecewise-constant component, scan its weight on the full dataset and retain
fits that:

1. improve reconstruction enough to justify including the component;
2. contain no more than a plausible number of jumps;
3. contain no jump below a meaningful effect size.

Then choose the least-regularized admissible fit. Reconstruction improvement is
a gate, not the quantity being optimized; jump count and minimum jump magnitude
define structural credibility. The same pattern generalizes: translate what a
human would reject in the component into explicit admissibility checks, then
select among the surviving fits.

The rule must admit a **null result**. If no candidate satisfies the structural
acceptance checks, select the most-regularized, no-event model rather than
forcing a detection. In an event detector, the minimum effect size is the
sensitivity control: raising it suppresses false positives but misses smaller
real events; lowering it increases sensitivity at the cost of more false
positives. Treat it as a domain specification, not a fit-derived weight.

See the soiling example in [formulation.md](formulation.md) and the IRL1
discussion in [component-catalog.md](component-catalog.md) for other cases where
valid, `optimal` fits still require structural inspection.

There is no universal automated Tier-3 criterion. Record the structural rule
and its rationale rather than disguising it as a generic fit metric.

## Simplify when the tiers do not separate

Many interacting Tier-2 knobs signal an over-coupled specification. Remove weak
components, constrain ambiguous ones, or return to exploration. Parsimony means
minimizing unresolved interaction, not merely component count. Close with a
critique of uncertain forms, flat holdout choices, and unresolved structural
judgments.

## Handoff to implementation

Write down:

- each component and functional form;
- every hyperparameter, its tier, chosen value, and selection evidence;
- the time grid, Δ, transform, mask policy, and residual loss;
- unresolved structural assumptions and their domain owner;
- whether a separate tuning dataset exists.

Tier-1 values and Tier-2 values selected on a representative tuning set can
ship as constants. Without a tuning set, a Tier-2 selector may need to run on
each production batch. A Tier-3 rule must travel with the model whenever new
data can change the structural feature being measured.

Run bootstrap confidence intervals only after the operating model has been
selected for the dataset. Bootstrapping inside the tuning loop mixes model
selection variation with final-model uncertainty. See
[implementation.md](implementation.md) for repeated-solve mechanics.
