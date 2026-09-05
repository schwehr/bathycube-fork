# AGENTS.md

Guidance and instructions for AI agents working in this repository.

## Project Overview

`bathycube` is a Python implementation of the Combined Uncertainty and Bathymetry Estimator (CUBE)
gridding algorithm for bathymetric data, originally developed at the Center for Coastal and Ocean
Mapping / NOAA/UNH Joint Hydrographic Center (CCOM/JHC).

The repository provides two implementations:

- `bathycube/cube.py`: Pure Python implementation allowing inspection and interactive manipulation
  of node and grid objects.
- `bathycube/numba_cube.py`: High-performance Numba JIT-compiled implementation (~55x faster) using
  `jitclass` and `@njit` kernels.

## Environment & Tooling

- **Python Version**: `>=3.14`
- **Package & Dependency Manager**: `uv`
- **Build System**: `setuptools`
- **Linters & Formatters**:
  - `ruff` (formatting and linting, line length 100)
  - `ty` (type checking)
  - `pyrefly` (type checking)
  - `pylint` (static analysis, config in `pyproject.toml`)
  - `codespell` (spell checking)
  - `mdformat` (markdown formatting, wrap 100)
- **Git Commit Style**: Conventional Commits (enforced by `conventional-pre-commit`)

## Common Commands

### Running Tests

```bash
uv run pytest
```

Coverage is automatically reported via `pytest-cov` as configured in `pyproject.toml`:

```bash
uv run pytest --cov=bathycube --cov-report=term-missing
```

### Pre-commit & Quality Checks

Pre-commit runs all linters, formatters, type checkers, and test hooks:

```bash
uv run pre-commit run --all-files
```

Individual checks can be run directly:

```bash
uv run ruff format .
uv run ruff check .
uv run pylint bathycube tests
uv run ty
uv run pyrefly check
```

## Architecture & Code Conventions

### Dual Implementation Parity

- Algorithms in `bathycube/cube.py` and `bathycube/numba_cube.py` mirror each other.
- Any mathematical or algorithmic changes to one must be reflected in the other and tested via
  `tests/test_compare_numba_base.py`.

### Numba Constraints

- `bathycube/numba_cube.py` relies on Numba `jitclass` (`cube_params_spec`, etc.) and `@njit`.
- Python objects and dynamic dictionaries cannot be passed into jitted functions directly without
  adhering to Numba-compatible types (`numbaf32`, `numbaf64`, `numbastr`, `numbai32`, etc.).
- When writing tests for jitted functions without JIT compilation overhead, use `.py_func` to test
  underlying Python functions directly.

### Type Annotations

- Provide precise, tight type annotations for all function signatures, return types, and class
  instance attributes using modern Python (3.14+) syntax.
- Avoid generic `Any`; prefer specific types such as `Sequence[int]`, `Buffer`, `Self`, or
  `Literal`.
- Avoid explicit `Union`/`Optional` types; use `|` (e.g. `X | None`).
- Use standard container types directly (e.g. `tuple[float, float]`, `list[str]`, `dict[str, Any]`).

### Code & Docstring Style

- **Docstrings**:
  - **CRITICAL RULE**: All module, class, method, and function docstrings must strictly follow
    **Standard Google Python Docstring Style**.
  - Include clearly formatted `Args:`, `Returns:`, `Raises:`, `Yields:`, and `Attributes:` sections
    as applicable.
  - Avoid unstructured, verbose, or legacy docstring formatting.
- **String Formatting**:
  - Always use modern Python **f-strings** (`f"Value: {val}"`) for string concatenation and
    formatting. Never use legacy `%` formatting or `.format()`.

### Pylint & Code Health

- Warnings disabled in `pyproject.toml` are being progressively fixed and re-enabled.
- When fixing a class of warnings, clean up the codebase and re-enable the check by removing it from
  the `disable` list in `pyproject.toml`.

## Git & Review Guidelines

### Code Review

- Always perform a code review before committing. In addition to finding and suggesting fixes to
  issues, try to create 1-3 suggestions for improvement to the code based on the current changes.
- Check if `AGENTS.md` needs updates based on the current changes and propose improvements.

### Conventional Commits

- All git commit messages MUST follow the
  [Conventional Commits](https://www.conventionalcommits.org/) specification:
  - Format: `<type>(<optional scope>): <description>`
  - Examples:
    - `feat(dunder): enable modern __add__ and __iadd__ support`
    - `refactor(cube): simplify if statement in has_nomination`
    - `refactor(tests): switch test_init.py from unittest to pytest`
    - `style: format code with ruff and set line-length to 100`
    - `test(numba_cube): add coverage for hypothesis extraction`
    - `build: enable import-error in pylint configuration`
    - `chore(license): replace __copyright__ variable with SPDX header`
    - `docs: import legacy manuals into docs/ directory`

### NO Tag or Conversation ID Entries

- **CRITICAL RULE**: Commit messages must **NEVER** contain `TAG=` or `CONV=` lines or entries.
  These are reserved for internal Piper/CL tools and must be omitted from all git commits in this
  repository.
