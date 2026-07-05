# PACE 2026 Heuristic MAF Solver

Solver for the PACE 2026 heuristicntrack problem on rooted maximum agreement forests.

## Build
Build from the repository root:

```sh
make
```

This creates:

```sh
./pace_solver
```

For a Debian container or fresh Debian system, `docker_setup.sh` installs the
needed packages and runs the build.

## Run

The solver reads one PACE instance from standard input and writes a forest to
standard output:

```sh
./pace_solver < instance.txt > solution.txt
```

Optional local profiling:

```sh
./pace_solver --profile-dir profiling/run_name < instance.txt > solution.txt
```

