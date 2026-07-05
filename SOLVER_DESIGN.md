# Solver Design

This document explains the current solver in enough detail for someone to
understand, profile, tune, or extend it. It describes the mathematical model,
the internal representation, every major source file, all reductions and
heuristics, and the top-level anytime schedule.

The solver is built as one native C++ executable, `pace_solver`, from
`src/main.cpp` and the headers under `src/`.

## 1. Problem Model

The PACE 2026 heuristic-track instance used by this solver contains exactly two
rooted binary phylogenetic trees on the same taxon set. Taxa are positive
integer labels. The output is an agreement forest: a partition of the taxon set
into components such that each component induces the same rooted topology in
both input trees and such that the induced edge sets of different components
are edge-disjoint in both trees.

The optimization goal is to minimize the number of forest components.

Let:

- `X` be the taxon set, `|X| = n`.
- `T1` and `T2` be the two rooted binary input trees.
- `F = {C1, ..., Ck}` be a partition of `X`.
- `T|C` be the rooted tree induced by labels `C`, after suppressing degree-2
  nodes.

Then `F` is valid when:

1. `C1, ..., Ck` are nonempty, disjoint, and cover `X`.
2. For every component `C`, `T1|C` and `T2|C` have the same rooted labelled
   topology.
3. For every pair of components `A != B`, the paths used by `T1|A` and `T1|B`
   are edge-disjoint in `T1`; likewise in `T2`.

The solver minimizes `k`.

Equivalently, if a candidate component `C` has `|C|` labels, selecting `C`
saves `|C|-1` components relative to leaving those labels as singletons. Thus
set-packing based phases maximize:

```text
sum_{selected C} (|C| - 1)
```

subject to label-disjointness, topology compatibility, and edge-disjointness in
both trees. The final component count is:

```text
n - sum_{selected C} (|C| - 1)
```

This equivalence is central to `AgreementComponentPacking` and to several
local candidate generators.

## 2. Source Tree Overview

```text
src/
  core/
    Tree.hpp
    Forest.hpp
    Timer.hpp
    Random.hpp
  exact/
    TinyExactOracle.hpp
  heuristics/
    FastCherryApprox.hpp
    CherryPairApprox.hpp
    ActiveCherryGreedyApprox.hpp
    AgreementComponentPacking.hpp
    ForestCrossover.hpp
    LocalImprove.hpp
  io/
    NewickParser.hpp
    PaceParser.hpp
  reductions/
    CommonSubtreeReduction.hpp
    ChainReduction.hpp
    ThreeTwoChainReduction.hpp
    ClusterDecomposition.hpp
    ReductionStack.hpp
  main.cpp
```

The solver is mostly header-only, with `main.cpp` orchestrating parsing,
validation, reductions, scheduling, profiling, output safety, and the portfolio
of heuristics.

## 3. Core Data Structures

### 3.1 `Tree`

Defined in `src/core/Tree.hpp`.

`Tree` is the solver-friendly representation of a rooted binary tree. It is
converted from the parsed Newick tree and stores flat arrays:

- `nodes`: node records containing `left`, `right`, and leaf `label`.
- `root`: integer node index of the root.
- `parent`: parent node per node.
- `depth`: root depth per node.
- `tin`, `tout`: DFS intervals for ancestor checks.
- `postorder`: postorder traversal.
- `leaf_count_under`: number of leaves under each node.
- `leaf_labels`: sorted taxon labels.
- `leaf_to_node_flat`: fast dense label-to-node lookup.
- `up`: binary lifting table for LCA.

Important operations:

- `node_of_label(x)`: returns the leaf node for label `x`.
- `is_ancestor(a, b)`: tests whether `a` is an ancestor of `b` using DFS
  intervals.
- `lca(a, b)`: lowest common ancestor in `O(log n)`.
- `labels_under(u)`: sorted labels below a node.

The code uses this representation because MAF heuristics need many repeated LCA
queries, path extractions, and leaf lookups. Flat arrays avoid pointer chasing
and help cache locality.

### 3.2 `LabelForest` and `LabelComponent`

Defined in `src/core/Forest.hpp`.

The solver stores a forest as label blocks rather than as explicit tree
fragments:

```cpp
struct LabelComponent {
    std::vector<std::uint32_t> labels;
};

struct LabelForest {
    std::vector<LabelComponent> components;
};
```

`normalize()` sorts labels inside components, removes duplicates, drops empty
components, and sorts components by first label. `validate_partition_of()`
checks that a forest covers exactly a given label set with no duplicates.

This label-block representation is deliberate. It makes crossover, packing, and
reduction expansion simpler. Actual Newick output is reconstructed at the end
by inducing each label block from the original first tree.

### 3.3 Timers

Defined in `src/core/Timer.hpp`.

`Timer` stores a start time and a deadline. All expensive phases use
`should_stop(guard_seconds)` before starting work and inside loops. This lets
the solver stop before the external timeout and preserve time to publish.

The top-level process uses:

- an external 300-second default process timer;
- a 30-second baseline timer sharing the process start time;
- an extended timer ending at `external_limit - 5` seconds.

### 3.4 Randomness

Defined in `src/core/Random.hpp`.

`Random` is a deterministic SplitMix-style generator seeded by
`SolverConfig::seed` by default. It is used for controlled shuffling in local
improvement and for salted active-cherry portfolios.

## 4. Parsing and Output

### 4.1 Newick Parsing

`src/io/NewickParser.hpp` parses rooted binary Newick trees with integer labels.
Internal node ids are assigned during parsing. Leaf ids equal leaf labels.

### 4.2 PACE Parsing

`src/io/PaceParser.hpp` reads:

- `#p` header;
- optional STRIDE metadata lines;
- optional approximation target line;
- optional parameter lines;
- Newick trees.

The solver currently requires exactly two rooted trees.

### 4.3 Output Reconstruction

The solver outputs one component per line. For each component label set:

1. If the component has one label, print the label.
2. If it has two labels, print a small cherry-like Newick expression.
3. Otherwise, induce the component from the original first tree by recursively
   retaining only selected labels and suppressing missing branches.

The output is valid because every accepted forest is validated as an agreement
forest against both original trees before it can become the global incumbent.

## 5. Agreement-Forest Validation

Validation in `main.cpp` has three layers.

### 5.1 Partition Check

The forest must contain every original label exactly once.

### 5.2 Restricted Topology Check

For a component `C`, the solver builds a virtual induced tree signature for
`T1|C` and `T2|C`.

The implementation:

1. Converts labels to leaf nodes.
2. Sorts them by DFS order.
3. Adds LCAs of adjacent selected leaves.
4. Sorts and unique-ifies the virtual node set.
5. Builds parent relationships by an ancestor stack.
6. Interns unordered rooted child signatures.

If the two signatures match, the restricted topologies are considered equal.
Components of size 1 or 2 are always topology-compatible.

### 5.3 Edge-Disjointness Check

For each component and each tree, the induced edge footprint is the union of
paths from each selected leaf to the component LCA. Edges are represented by the
child node index below the edge. A forest is edge-disjoint if no edge is owned
by two components.

This same edge footprint idea is reused by local improvement, packing,
crossover, and component-collapse exactification.

## 6. Incumbent Safety and Tie-Breaking

`consider_candidate()` is the central acceptance gate.

A candidate replaces the incumbent only if:

1. It has fewer components; or
2. it has equal components and a larger tie score; and
3. it validates as an agreement forest.

The tie score is:

```text
sum_C |C|^2
```

This favors structurally larger components at equal component count. Equal
component forests are useful as seeds, but the published incumbent uses this
simple deterministic tie-break.

## 7. Emergency Publishing

The solver must be ready when STRIDE/Optil sends `SIGTERM`. The top-level code
keeps a serialized valid forest in global emergency storage.

At startup:

```text
emergency output = singleton forest
```

Whenever a better validated global forest is found:

```text
serialize forest
atomically replace emergency output pointer/size
```

The signal handler for `SIGTERM` and `SIGALRM` writes the current emergency
buffer directly to stdout using `write()` and exits. This means the solver
should not lose score merely because it was inside a long phase when the time
limit arrived.

The final normal path also prints the best incumbent with `write_forest()`.

## 8. Reductions

Reductions run in `apply_reductions_exhaustively()`. The solver repeats passes
until no rule changes the instance, the pass limit is reached, or time runs
short.

The reduction stack records enough information to expand a reduced forest back
to original labels.

### 8.1 Common Subtree Reduction

Defined in `src/reductions/CommonSubtreeReduction.hpp`.

A rooted subtree with exactly the same labelled topology in both input trees can
be contracted to one placeholder taxon. The solver computes canonical subtree
signatures bottom-up:

```text
sig(leaf x) = leaf(x)
sig(internal a,b) = internal(min(sig(a), sig(b)), max(sig(a), sig(b)))
```

Candidate common subtrees are matched by signature and leaf count. A selected
common subtree is replaced by a placeholder label in both trees. Expansion later
replaces the placeholder with all original labels in the same output component.

This preserves feasibility because a common subtree can stay intact in some
optimal forest.

### 8.2 Chain Reduction

Defined in `src/reductions/ChainReduction.hpp`.

A common rooted chain is a sequence of taxa occurring along the same pendant
chain structure in both trees. The solver keeps the first three labels and
removes the suffix. During expansion:

- if the kept prefix is still in one component, the removed suffix is reattached
  to that component;
- otherwise the removed labels are restored as singletons.

The singleton fallback is always feasible but is not certification-preserving,
because the stack alone does not prove that no better placement exists in that
case. The profile logs expose this through `reduction_certification_preserving`
and related counters.

### 8.3 3-2 Chain Reduction

Defined in `src/reductions/ThreeTwoChainReduction.hpp`.

This rule detects a local 3-chain versus 2-chain pattern and deletes one taxon.
Expansion adds the deleted label back as a singleton. This is conservative and
keeps feasibility.

### 8.4 Cluster Decomposition

Defined in `src/reductions/ClusterDecomposition.hpp` and orchestrated in
`solve_instance()`.

A common cluster is a subset of labels that forms a rooted cluster in both
trees. The cluster splitter can produce four subinstances:

- bottom without marker;
- bottom with `X_DOWN`;
- top without the cluster;
- top with `X_UP`.

The current solver computes a closed state and can optionally compute an open
state:

- `closed`: the cluster boundary is cut; bottom and top solutions are united.
- `open`: marked bottom and marked top are solved, marker components are
  merged after removing `X_DOWN` and `X_UP`, then validated.

The open state corresponds to the case where one bottom component and one top
component bridge the cluster boundary. The solver keeps the state with the
lower component count when available. Open-state solving is gated by cluster
size, side size, depth, elapsed time, and remaining time because solving marked
subproblems everywhere is too expensive.

The hierarchy lookup cache avoids recomputing common-cluster templates when the
same reduced pair is seen repeatedly.

## 9. Exact Tiny Oracle

Defined in `src/exact/TinyExactOracle.hpp`.

For very small instances, the solver can enumerate all subsets of labels and
solve the exact set-packing problem.

For `n <= max_leaves`:

1. Enumerate every nonempty subset mask.
2. Keep masks whose induced topology is equal in both trees.
3. Compute edge masks in both trees.
4. Search for a minimum number of compatible masks covering all labels.

The DFS branches on the first uncovered label. It prunes by current best count,
maximum candidate size, timer, and node limit. Results are memoized by a
canonical pair key.

This is used only for tiny whole instances and tiny cluster subproblems because
subset enumeration is exponential.

## 10. Heuristic Engines

### 10.1 Fast Common-Subtree Heuristic

Defined in `src/heuristics/FastCherryApprox.hpp`.

This is the fastest strong baseline. It computes canonical rooted-subtree
signatures in both trees and selects large matching subtrees as components. It
then covers remaining labels by singletons.

It is safe because each selected component is a common rooted subtree and
selected subtrees are disjoint.

### 10.2 Cherry Pair Heuristic

Defined in `src/heuristics/CherryPairApprox.hpp`.

This heuristic collects cherries from both trees. A cherry pair `{a,b}` is a
size-2 agreement component. Candidates are sorted by induced edge cost and
greedily selected subject to:

- labels unused;
- induced edges unused in both trees.

Uncovered labels become singletons.

### 10.3 Active Cherry Greedy

Defined in `src/heuristics/ActiveCherryGreedyApprox.hpp`.

This is one of the two main quality engines.

It maintains two dynamic trees:

- `active_t1`: active tree where common cherries are contracted;
- `forest_t2`: a dynamic forest whose cuts define the output components.

The loop:

1. Contract all currently common cherries.
2. Enumerate active cherries in `active_t1`.
3. Score candidate operations according to a policy.
4. Apply a cut/merge action in `forest_t2`.
5. Repeat until the active tree is small, no usable cherry remains, or time
   runs out.

Policies include:

- `Balanced`
- `PreferSmallCuts`
- `PreferBigProgress`
- `PreferDifferentComponent`
- `PreferLowConflictMass`
- `PreferFewPendants`
- `PreferImmediateGain`
- `ConservativeSingleCut`
- `DualityConservative`
- `ResolveFinalCut`
- `AggressiveMultiCut`

Portfolio diversity comes from policy choice, swapped tree order, sample caps,
salted sampling offsets, and small discrepancy scripts that intentionally take
the second-ranked action at selected steps.

The output is validated as a label forest.

### 10.4 Agreement Component Packing

Defined in `src/heuristics/AgreementComponentPacking.hpp`.

This is the other main quality engine. It treats MAF as a restricted weighted
set-packing problem over candidate agreement components.

For each candidate component `C`, store:

- sorted labels;
- leaf ids;
- induced edge footprint in `T1`;
- induced edge footprint in `T2`;
- gain `|C|-1`;
- edge cost;
- source flags and tie hash.

A selected set must be:

- label-disjoint;
- edge-disjoint in `T1`;
- edge-disjoint in `T2`.

Candidate sources:

- components from seed forests;
- extra candidate components supplied by main;
- intersections of components from seed forests;
- all/pseudo-random pairs;
- small extensions around tree order;
- structure-driven subtree and neighborhood growth;
- archive components when enabled.

Solving layers:

1. Seed state: keep the best complete seed as an initial state.
2. Greedy portfolio: multiple candidate orderings select compatible components.
3. Exchange search: tries to remove selected components and refill with better
   alternatives.
4. Deficit-refill search: controlled ejection/refill moves with bounded
   deficit.
5. Local MWIS repair: solves small conflict neighborhoods exactly.
6. Bounded exact search: for small candidate pools/leaves, solve a restricted
   exact master.

If selected candidates cover only part of the label set, all uncovered labels
become singletons.

The packing objective is:

```text
maximize sum_C (|C| - 1)
```

which is equivalent to minimizing final component count.

### 10.5 Local Improve

Defined in `src/heuristics/LocalImprove.hpp`.

This is a conservative merge-only local search. It can try:

- singleton reinsertion into a larger component;
- pair merges;
- structural pair merges between nearby components.

Each tentative merge must pass topology and edge compatibility. It is useful as
a cheap cleanup phase but is not a full destroy-and-repair search.

### 10.6 Exact Pairwise Forest Crossover

Defined in `src/heuristics/ForestCrossover.hpp`.

Given two complete valid forests `F` and `G`, components inside `F` are mutually
compatible and components inside `G` are mutually compatible. Therefore conflict
edges among non-singleton components only run between `F` and `G`. The conflict
graph is bipartite.

Each component has weight:

```text
w(C) = |C| - 1
```

The goal is maximum-weight independent set in the bipartite conflict graph.
This is solved exactly through minimum vertex cover via a min cut:

```text
source -> F component, capacity w(C)
G component -> sink, capacity w(C)
F component -> G component, capacity INF for each conflict
```

The complement of the min vertex cover is the maximum-weight independent set.
Uncovered labels become singletons.

Shared components are deduplicated before building the bipartite part.

## 11. Main-Level Candidate Generators

These live in `src/main.cpp` because they combine multiple engines and use
global scheduling information.

### 11.1 Component and Forest Archive

`ComponentForestArchive` stores:

- diverse valid forests;
- unique non-singleton component label sets;
- component support counts and last-use epochs.

It is enabled for `220 <= n <= 5000`. The archive feeds packing and crossover
with diverse candidates without regenerating every component from scratch.

### 11.2 Incumbent Union Repacking

`generate_incumbent_union_components()` creates candidate components by merging
nearby incumbent components in tree order. It uses:

- component roots in both trees;
- DFS order in both trees;
- sliding windows;
- neighbor pairs;
- size caps.

The generated label sets are passed to strict packing. This targets situations
where the incumbent is close but several adjacent components can be collapsed.

### 11.3 Singleton Ejection/Repacking

`generate_singleton_rescue_components()` and related seed-shatter/seed-union
helpers generate candidates around stubborn singleton labels.

Important sources:

- attach singleton `x` to high-support seed/incumbent components;
- attach two singletons to a base component;
- swap out one or more labels from a base component to insert a singleton;
- shatter blocked seed components by removing bad owner groups;
- union nearby seed components.

The generated candidates are fed to strict packing, so acceptance requires a
validated strict component reduction.

### 11.4 Component-Collapse Exactifier

`run_component_collapse_exactifier()` targets near-best small and medium cases.

It generates unions of current incumbent components:

- singleton-centered pairs;
- singleton plus two nearby components;
- small windows in `T1` and `T2` order;
- up to bucket-dependent caps on labels, candidates, and attempts.

For each candidate union, it checks:

1. the merged label set has identical induced topology in both trees;
2. the merged edge footprint conflicts only with components being removed;
3. the full reconstructed forest validates.

If no direct collapse is accepted, small cases can feed the best collapse
candidates into a strict restricted packing pass.

This phase is incumbent-safe: it can only publish a strictly smaller validated
forest.

### 11.5 Reduced-Global Candidate Portfolio

`run_reduced_global_candidate_portfolio()` creates alternate complete forests
by:

1. copying the original trees;
2. reapplying reductions independently;
3. running active-cherry with special policies such as `ResolveFinalCut` or
   `DualityConservative`;
4. expanding reductions back to the original label set;
5. validating and using the result as a candidate/seed.

This is particularly useful on large instances because reductions can expose
different active-cherry choices than the unreduced main run.

### 11.6 Reduced-Global Booster

`run_reduced_global_booster()` is an extended-stage broad reducer. It runs
salted variants of reduced-global active-cherry:

- `ResolveSet`;
- `DualitySeed`;
- `BalancedSeed`;
- swapped and unswapped tree order;
- different salts and step multipliers.

Valid produced forests are:

1. tested directly against the incumbent;
2. immediately crossed with the incumbent;
3. retained as seeds;
4. repacked using `AgreementComponentPacking`.

This phase is meant to turn unused 300-second time into diverse, globally
different candidate forests.

### 11.7 Large Balanced Seed Bank

`run_large_balanced_seed_bank()` runs salted `Balanced` active-cherry variants
on large instances. The resulting seed bank is used for:

- direct incumbent comparison;
- crossover;
- final global repacking;
- singleton/ejection repacking.

## 12. Top-Level Schedule

The solver is anytime. It always keeps a valid emergency solution and only
updates the incumbent through validation.

### 12.1 Startup

1. Parse the PACE instance.
2. Build original core trees.
3. Set emergency output to singleton forest.
4. Open profile logging if `--profile-dir` is set.
5. Install `SIGALRM` and `SIGTERM` emergency handler.
6. Initialize the global incumbent as singleton forest.

### 12.2 Warm Start

`make_initial_heuristic_result()` runs immediately:

1. `FastCherryApprox`
2. `CherryPairApprox`
3. active-cherry portfolio in baseline mode
4. normal agreement-component packing

Every improvement updates the emergency output.

### 12.3 Baseline Stage, Targeting 30 Seconds

The baseline timer is capped at 30 seconds. This stage is intended to produce a
strong solution even if the competition runner kills at 30 seconds.

If `n >= direct_cheap_output_min_leaves` (default 4000), the solver uses the
warm result directly rather than running full recursive solve. This protects
large instances from spending the 30-second budget in reductions/recursion.

Otherwise it runs `solve_instance()` with baseline config.

After that:

1. Early archive crossover if archive has enough forests.
2. Reduced-global candidate portfolio, mostly for large direct cases.
3. Forest crossover on reduced-global seeds.
4. Global repacking with reduced-global seeds.
5. Incumbent-union repacking for `n < 2500`.
6. Component-collapse exactifier for `n <= 2500`.
7. Singleton-ejection repacking for small/medium cases when time remains.
8. Validate and publish baseline incumbent.

### 12.4 Extended Stage, Targeting 300 Seconds

The extended stage runs only if the external limit exceeds the baseline plus
output reserve. Its deadline is `external_limit - 5 seconds`.

Extended config changes:

- enables deeper/smaller cluster recursion;
- raises local-improve max leaves;
- disables direct cheap cutoff;
- expands cluster-open profile limits;
- enables singleton rescue only for `n <= 220`.

Extended schedule:

1. Add baseline forest to final seed set for large instances.
2. Early incumbent-union repacking for `n >= 6000`.
3. Extended reduced-global portfolio.
4. Crossover on extended reduced-global seeds.
5. Repacking on extended reduced-global seeds.
6. Reduced-global booster for `n >= 1000`.
7. Reserve a final-global packing budget for large instances.
8. Run full `solve_instance()` again with extended config.
9. Final forest crossover.
10. Extended component-collapse exactifier for `n <= 5000`.
11. Late cluster-open retry for eligible medium instances.
12. Final extended-global repacking for `n >= 2500`.
13. Late incumbent-union repacking for `n >= 6000`.
14. Extended singleton-ejection repacking for `n < 6000`.
15. Large balanced seed bank, crossover, repacking, and large ejection.
16. Reserved-tail 5 percent bucket repacking for `n >= 6000`.
17. Final validation and output.

The small alternate exactification tail exists in code but is currently disabled
by `use_small_exactification_tail = false`.

### 12.5 Recursive `solve_instance()`

`solve_instance()` performs:

1. Exhaustive reductions.
2. Tiny exact oracle if reduced size is tiny.
3. Cluster split if a common cluster is found and policy says to split.
4. Closed cluster solve, optional open cluster solve, choose better state.
5. Otherwise `solve_without_cluster_recursion()`.
6. Expand reductions.
7. Archive and publish validated reduced-progress expansions.

### 12.6 `solve_without_cluster_recursion()`

This runs on a reduced pair without further cluster decomposition:

1. Fast cherry.
2. Cherry pair.
3. Active-cherry portfolio.
4. Agreement-component packing.
5. Early return on huge baseline cases or time guard.
6. LocalImprove for eligible sizes.
7. Post-restart packing for `220 < n < 600`.
8. Validate, archive, publish, return.

## 13. Packing Configuration

`agreement_packing_options()` is heavily size- and mode-dependent.

Modes:

- `Normal`: ordinary packing.
- `FinalGlobal`: larger candidate universe and longer local search.
- `ReservedTail5`: large-instance tail biased toward bigger generated
  components.
- `ReservedTail10` and `ReservedTail15`: still defined, but the current
  schedule uses only the 5 percent reserved tail.

Important knobs:

- `max_candidates`
- `order_window`
- `random_partners_per_leaf`
- `extension_neighbors`
- `max_generated_component_size`
- `structural_subtree_max_size`
- `structure_growth_max_component_size`
- `exchange_rounds`
- `two_exchange_pool`
- `local_mwis_anchors`
- `enable_exact`
- `local_time_limit_seconds`

`packing_focused` increases candidate and search budgets, especially for medium
and large final-global phases.

## 14. Profiling

Pass:

```sh
--profile-dir DIR
```

Each instance writes a JSONL profile file:

```text
DIR/profile_n<N>_<hash>.jsonl
```

Events include:

- `instance_start`
- `phase_result`
- `phase_skipped`
- `candidate_bank`
- `reduction`
- `cluster_split`
- `cluster_profile_decision`
- `main_final`

`phase_result` contains:

- phase name;
- duration;
- before/after component count;
- accepted flag;
- phase-specific counters.

Packing phases additionally expose detailed candidate counts, rejection counts,
exchange stats, exact stats, and selected-size distribution.

## 15. Mathematical Details Used Repeatedly

### 15.1 Induced Topology Signatures

For a label set `C`, the induced rooted topology can be represented by a
canonical unordered binary signature. In a rooted phylogenetic tree where child
order is irrelevant:

```text
sig(leaf x) = x
sig(node with children a,b) = pair(min(sig(a), sig(b)), max(sig(a), sig(b)))
```

For a restricted label set, the solver builds the virtual induced tree from
selected leaves and their LCAs. Equal signatures in both trees imply the same
restricted rooted topology.

### 15.2 Edge Footprints

For labels `C`, let:

```text
r_T(C) = LCA_T(C)
E_T(C) = union over x in C of path edges from x to r_T(C)
```

Two components `A` and `B` are edge-compatible in tree `T` when:

```text
E_T(A) intersection E_T(B) = empty
```

A valid forest requires this in both `T1` and `T2`.

### 15.3 Set Packing Formulation

Let `P` be a candidate set of valid agreement components. A subset `S` is
feasible if:

- no label appears in two selected components;
- no induced edge in `T1` appears in two selected components;
- no induced edge in `T2` appears in two selected components.

The restricted master problem is:

```text
maximize sum_{C in S} (|C| - 1)
subject to label and edge resources used at most once.
```

Given `S`, uncovered labels become singleton components. Thus maximizing saved
components minimizes output component count over the candidate universe.

### 15.4 Bipartite Crossover Exactness

For two complete valid forests `F` and `G`, conflicts among candidate
components only occur between `F` and `G`. The maximum saved weight is
maximum-weight independent set in a bipartite graph. The solver reduces this to
minimum vertex cover via min cut:

```text
source -> F component, capacity |C|-1
G component -> sink, capacity |C|-1
F component -> G component, capacity INF for each conflict
```

The selected independent set is the complement of the minimum vertex cover.

## 16. File-by-File Responsibilities

### `src/main.cpp`

Owns:

- command-line parsing;
- profiling;
- emergency output;
- validation;
- reduction orchestration;
- cluster recursion;
- schedule;
- archive;
- candidate generators;
- packing/crossover wrappers;
- final output.

This file is the control plane.

### `src/core/Tree.hpp`

Array-based rooted binary tree representation with parent/depth/DFS/LCA
precomputation.

### `src/core/Forest.hpp`

Label-block forest representation, normalization, and partition checks.

### `src/core/Timer.hpp`

Global and local time-budget checks.

### `src/core/Random.hpp`

Deterministic SplitMix-style random utility.

### `src/io/NewickParser.hpp`

Recursive parser for integer-labelled rooted binary Newick trees.

### `src/io/PaceParser.hpp`

PACE instance parser, including metadata and optional STRIDE fields.

### `src/reductions/CommonSubtreeReduction.hpp`

Canonical-signature common-subtree detection and contraction.

### `src/reductions/ChainReduction.hpp`

Common rooted-chain detection and suffix deletion.

### `src/reductions/ThreeTwoChainReduction.hpp`

3-2 chain pattern detection and safe singleton restoration.

### `src/reductions/ClusterDecomposition.hpp`

Common-cluster detection and construction of bottom/top open/closed
subinstances.

### `src/reductions/ReductionStack.hpp`

Reverse expansion of reduced forests back to original label sets and
certification-preservation reporting.

### `src/exact/TinyExactOracle.hpp`

Memoized exact search for tiny reduced instances.

### `src/heuristics/FastCherryApprox.hpp`

Fast common-subtree based heuristic.

### `src/heuristics/CherryPairApprox.hpp`

Greedy cherry-pair packing heuristic.

### `src/heuristics/ActiveCherryGreedyApprox.hpp`

Dynamic active-cherry heuristic with multiple scoring policies.

### `src/heuristics/AgreementComponentPacking.hpp`

Candidate generation and restricted set-packing engine.

### `src/heuristics/ForestCrossover.hpp`

Exact pairwise forest crossover via bipartite min cut.

### `src/heuristics/LocalImprove.hpp`

Conservative merge-only local improvement.

## 17. Safety Invariants

The solver relies on several invariants:

1. The emergency output is always syntactically complete and valid.
2. Global incumbent updates go through `consider_candidate()`.
3. Every externally published improvement is validated against the original
   trees.
4. Reductions are expanded before candidates are compared to the original
   incumbent.
5. Time guards are checked before and during expensive phases.
6. Exceptions inside heuristic phases are caught and treated as "no
   improvement".

## 18. Known Limitations

The solver is a heuristic solver with some exact subroutines, not a complete
exact MAF solver for large instances.

Important caveats:

- `AgreementComponentPacking` is exact only over the candidate universe it has
  generated.
- Tiny exact search is limited to very small reduced instances.
- Chain expansion can fall back to singleton restoration, which is feasible but
  not proof-preserving.
- Cluster open profiles are gated and are not solved at every split.
- LocalImprove is merge-only and cannot perform arbitrary destroy-and-repair.
- The small alternate exactification tail exists in code but is disabled in the
  current schedule.

## 19. Tuning Guidance

The highest-impact tuning areas are usually:

- active-cherry portfolio policy mix and sample caps;
- packing candidate budgets and generated-size quotas;
- final-global packing reserve for large instances;
- gates around reduced-global candidates and booster runs;
- archive enable range and seed caps;
- component-collapse caps for near-best cases;
- cluster open-profile size/depth/time gates.

When testing changes, compare both:

- total heuristic score;
- total component count;
- per-bucket regressions, especially `n <= 220`, `221..700`, `701..2500`,
  `2501..5000`, `5001..10000`, and `>10000`.

Because STRIDE runs instances in parallel, avoid phases that continue long after
the useful incumbent has stopped improving unless they have a strong profile
return. Long low-ROI phases can steal CPU from other concurrent instances.
