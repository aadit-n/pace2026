# Installation Guide

This repository builds the PACE 2026 heuristic-track solver as a single native
C++ executable named `pace_solver`.

## Requirements

Required external dependencies:

- Debian 13.5 or another Linux environment with a modern GNU toolchain.
- `g++` with C++17 support.
- `make`.
- POSIX threads support, provided by the system C/C++ runtime and linked through
  `-pthread`.

The solver itself does not require Boost, GMP, Python packages, or any external
C++ libraries.

Optional tools:

- `./stride`, if present, can validate solutions with `./stride check`.
- `pace26stride`/STRIDE tooling may be used for local benchmark runs, but is not
  required to build or run the solver.

## Build On Debian 13.5

From the repository root:

```sh
./docker_setup.sh
```

The script installs the required Debian packages and runs `make clean && make`.

Equivalent manual commands:

```sh
apt-get update
apt-get install -y --no-install-recommends build-essential ca-certificates make
make clean
make
```

The resulting executable is:

```sh
./pace_solver
```

## Running

The solver reads a PACE instance from standard input and writes a solution to
standard output:

```sh
./pace_solver < instance.txt > solution.txt
```

If the local STRIDE checker is available, validate with:

```sh
./stride check -q instance.txt solution.txt
```

## Notes

The default `Makefile` uses aggressive native optimization flags including
`-O3`, `-march=native`, and LTO. This is intended for competition performance on
the build machine. For a portable binary, remove or replace `-march=native` in
`CXXFLAGS` before building.
