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

## 3) Build STRIDE (if you do not already have `stride`)

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

## 4) Prepare an instance list (`.lst`)

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

## 5) Run solver with STRIDE

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

## 6) Check results and debug failures

STRIDE writes logs under `stride-logs/`.

- Latest run link: `stride-logs/latest/`
- Per-instance stderr/stdout: task folders under that run
- Machine-readable summary: `stride-logs/latest/summary.json`

If you see many `SolverError` results, check:
- solver path is executable: `--solver ./solver`
- solver exits with code `0` on success
- solver writes only solution to stdout (debug to stderr)
- shared libraries are resolvable at runtime

## 7) Minimal repeat workflow

```bash
cd genesis && make && cd ..
make
./pace26stride/target/release/stride run -i my_instances.lst --solver ./solver
```
