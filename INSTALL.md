# Installation Guide

This repository builds the PACE 2026 heuristic-track solver as a single native
C++ executable named `pace_solver`.

## Build On Debian 13.5

From the repository root:

```sh
./docker_setup.sh
```

The script installs the required Debian packages and builds `pace_solver`
directly with `g++`. It does not require `make` to be installed.

Equivalent manual commands:

```sh
apt-get update
apt-get install -y --no-install-recommends ca-certificates g++
rm -f pace_solver
g++ -O3 -march=native -flto -fno-plt -fno-rtti -fdevirtualize-speculatively \
    -Isrc -Isrc/io -Isrc/reductions \
    src/main.cpp -o pace_solver -pthread
```

If `make` is available, the repository `Makefile` can also be used:

```sh
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

## Notes

The default `Makefile` uses aggressive native optimization flags including
`-O3`, `-march=native`, and LTO. This is intended for competition performance on
the build machine. For a portable binary, remove or replace `-march=native` in
`CXXFLAGS` before building.
