# PACE 2026 Solver (C++) + Genesis + STRIDE Usage

This repository contains:
- `solver.cpp`: your C++ solver source
- `Makefile`: builds `./solver`
- `genesis/`: Genesis C++ library (used to parse Newick trees)
- `pace26stride/`: STRIDE runner source code

The folders `.vscode/`, `instances/`, `stride-downloads/`, `stride-logs/`, and the `solver` binary are local artifacts and are not required in git.

## 0) Prerequisites

Install basic build tools:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake zlib1g-dev cargo
```

If you already have `g++`, `cmake`, and Rust/Cargo installed, skip this.

## 1) Build Genesis

Your solver links against `libgenesis` from `genesis/bin`.

From repo root:

```bash
cd genesis
make clean
make
cd ..
```

This should produce shared libraries in `genesis/bin/` (for example `genesis/bin/libgenesis.so`).

## 2) Build the solver

From repo root:

```bash
make clean
make
```

This creates `./solver`.

## 3) Solver Function Contract (how to plug in your algorithm)

Your implementation should only modify `solve(const PaceInstance& instance)` in `solver.cpp`.

The runner code already does:
- parse stdin into a `PaceInstance`
- install SIGTERM handling
- print the current best known valid solution on timeout

### Input your `solve(...)` receives

`PaceInstance` has:

- `tree_count`: number of input trees (`#p t n` -> `t`)
- `leaf_count`: number of leaves (`#p t n` -> `n`)
- `trees`: `std::vector<genesis::tree::Tree>` of parsed Newick trees

The input has already been parsed and validated enough to populate these fields.

### Output your `solve(...)` must return

Return `std::vector<std::string>` where each element is one forest component in Newick format.

Examples of valid returned entries:
- `"(1,2);"`
- `"4;"`

Semicolon is optional in returned strings; the framework normalizes missing semicolons.

### Required correctness rules for returned forest

- Every leaf label from `1..leaf_count` appears exactly once across all returned components.
- No duplicate leaf labels across components.
- Components must be valid Newick trees.
- The forest should be an agreement forest for all input trees (checker enforces feasibility).

### SIGTERM / heuristic best-so-far behavior

You can publish intermediate improvements during search:

```cpp
publish_best_solution(current_best_forest);
```

Where `current_best_forest` is `std::vector<std::string>`.

If STRIDE sends SIGTERM, the process prints the last published solution immediately.

### Minimal solve template

```cpp
std::vector<std::string> solve(const PaceInstance& instance) {
    std::vector<std::string> best = build_singleton_forest(instance.leaf_count);
    publish_best_solution(best);  // optional extra safety while experimenting

    // TODO: your heuristic/exact logic here using instance.trees
    // If you find a better valid forest:
    // best = candidate;
    // publish_best_solution(best);

    return best;
}
```

### Practical design guidance

- Keep stdout strictly for solution lines only.
- Send debug/progress logs to stderr only.
- Maintain one always-valid `best` forest in memory.
- Only call `publish_best_solution(...)` with valid forests.

## 4) Build STRIDE (if you do not already have `stride`)

Build from the bundled source:

```bash
cargo build --release --manifest-path pace26stride/Cargo.toml
```

The executable will be:

```bash
./pace26stride/target/release/stride
```

Optional convenience alias:

```bash
alias stride=./pace26stride/target/release/stride
```

## 5) Prepare an instance list (`.lst`)

`stride run` requires `-i <list-or-instance...>`.

You have two common options:

### Option A: local instance files

Create `my_instances.lst` with one instance file path per line:

```text
path/to/instance1.nw
path/to/instance2.nw
```

### Option B: STRIDE idigests

Create `my_instances.lst` with entries like:

```text
s:0010b172a28d0664d5521e1296fc3586
```

If you use `s:<idigest>` entries, download them first:

```bash
./pace26stride/target/release/stride download -i my_instances.lst
```

This creates `stride-downloads/` automatically.

## 6) Run solver with STRIDE

Use the compiled binary (`./solver`), not `solver.cpp`.

```bash
./pace26stride/target/release/stride run -i my_instances.lst --solver ./solver
```

Useful flags:

```bash
./pace26stride/target/release/stride run -i my_instances.lst --solver ./solver -t 300 -g 5 -p 8
```

- `-t`: soft timeout seconds (SIGTERM sent)
- `-g`: grace seconds before SIGKILL
- `-p`: parallel jobs

## 7) Check results and debug failures

STRIDE writes logs under `stride-logs/`.

- Latest run link: `stride-logs/latest/`
- Per-instance stderr/stdout: task folders under that run
- Machine-readable summary: `stride-logs/latest/summary.json`

If you see many `SolverError` results, check:
- solver path is executable: `--solver ./solver`
- solver exits with code `0` on success
- solver writes only solution to stdout (debug to stderr)
- shared libraries are resolvable at runtime

## 8) Minimal repeat workflow

```bash
cd genesis && make && cd ..
make
./pace26stride/target/release/stride run -i my_instances.lst --solver ./solver
```
