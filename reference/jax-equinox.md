# AGENTS.md

Contract for coding agents working on JAX/Equinox scientific computing for statistical genetics.

> **Attribution:** The JAX/Equinox best practices, checklists, and code snippets in this document are adapted from [jax-numerics-agent](https://github.com/quattro/jax-numerics-agent/) by Nicholas Mancuso.

## Definitions (strict)

- **JIT boundary**: the public API function wrapped with `eqx.filter_jit`/`jax.jit`.
- **Static data**: non-array metadata that must be compile-time constant across calls; declare via `eqx.field(static=True)` or `eqx.partition`.
- **Dynamic data**: array leaves that change per-call and flow through JIT.
- **PyTree stability**: identical treedef and leaf shapes/dtypes across iterations and across calls; only values may change.
- **Abstract module**: `eqx.Module` containing `abc.abstractmethod` or `eqx.AbstractVar`; not instantiable.
- **Final module**: concrete `eqx.Module` with no further subclassing or overrides.
- **Deterministic**: same inputs and PRNGKey produce the same outputs under `jit/vmap/scan`.

## Non-negotiable rules (DO / DON'T)

**DO**

- Thread PRNGKeys explicitly through every stochastic function.
- Separate static config from dynamic data before JIT and keep it stable.
- Keep PyTree structure and shapes constant inside iterative loops.
- Use `lax.scan/while_loop/cond` for control flow under JIT.
- Validate critical assumptions (shape, dtype, finiteness) and surface failures predictably.
- Use `eqx.Module` for ABCs, with `AbstractVar` for abstract attributes.
- Define all fields and `__init__` in one class; concrete classes are final.

**DON'T**

- Don't capture JAX arrays in closures that cross `jit` or custom-AD boundaries.
- Don't use global RNG state or create keys inside jitted code.
- Don't mutate or replace PyTrees in ways that change structure mid-loop.
- Don't use Python `if/for/while` inside JIT when the condition depends on arrays.
- Don't raise Python exceptions inside JIT; use `eqx.error_if` or result codes.
- Don't subclass or override methods of a concrete module; use composition.
- Don't use `super()` in module hierarchies; follow abstract-or-final.
- Don't use `hasattr` for optional attributes; declare them on the ABC.

## Strong guidelines

- JIT public APIs once; avoid nested JITs in hot loops.
- Batch with `eqx.filter_vmap` and explicit `in_axes` for PyTrees and Modules.
- Prefer operator-based linear algebra over materializing large matrices.
- Use `jax.eval_shape`/`ShapeDtypeStruct` for structure validation.
- Choose a dtype policy up front; cast once to inexact types.
- Prefer x64 for stiff or ill-conditioned problems when accuracy matters.
- Use `__check_init__` (Equinox) to enforce invariants early.
- When ABC state varies, use `TypeVar`/`Generic[...]` to type solver state.

## Project engineering rules (software development)

**DO**

- Keep public API signatures stable; document any breaking change explicitly.
- Return structured results consistently (value + status/result + optional stats).
- Use `throw=False` modes where failures are expected and test them.
- Add docstrings for public APIs and Modules; use markdown sections like `**Arguments:**`/`**Returns:**` if the repo uses that style.
- Maintain CI gates: format, lint, typecheck, tests; do not merge with failing checks.
- Add regression tests for bug fixes and for AD/batching behavior.
- Record serialization/checkpoint formats and version any persisted state.
- Note determinism limits across devices/backends in docs when relevant.

**DON'T**

- Don't change return structure or error semantics without a deprecation period.
- Don't introduce silent behavior changes (e.g., new defaults) without docs and tests.
- Don't add logging in hot JIT paths; gate diagnostics behind flags.
- Don't serialize raw modules without documenting version/compat constraints.

## Performance reasoning

- Minimize retracing by isolating static arguments and keeping PyTree structures stable.
- Avoid shape-changing branches; use `lax.cond` and preserve output structure.
- If compile time dominates, stage large subgraphs (e.g., noinline-style wrappers).

## Numerics reasoning

- Guard divisions and norms; avoid NaNs/inf and define stable JVPs if needed.
- Prefer explicit `rtol/atol` and scaling choices; document the chosen norm.
- Detect nonfinite values early and propagate result codes explicitly.

## Shape semantics

- Treat shapes/dtypes as part of the API contract.
- Any change in PyTree structure is a breaking change and must be explicit.

---

# Checklists

## JAX/Equinox Design Checklist

Use this when defining APIs, Modules, and solver structure.

### Architecture

- [ ] Public APIs are the only JIT boundaries (`eqx.filter_jit` / `jax.jit`).
- [ ] Static configuration is separated and marked static (`eqx.field(static=True)` or `eqx.partition`).
- [ ] Dynamic inputs are arrays (or PyTrees of arrays) and are the only traced data.
- [ ] PyTree structure and shapes are stable across iterations and calls.

### Modules and ABCs

- [ ] Abstract classes are `eqx.Module` and contain `abc.abstractmethod` or `eqx.AbstractVar`.
- [ ] All fields and `__init__` live in a single class (no split initialization).
- [ ] Concrete modules are final (no overriding concrete methods, no subclassing).
- [ ] Optional attributes are declared on ABCs (no `hasattr` checks).
- [ ] If state types vary, use `TypeVar`/`Generic[...]` for solver state.
- [ ] `__check_init__` enforces invariants where appropriate.

### Control Flow

- [ ] Looping uses `lax.scan` (fixed length) or `lax.while_loop` (data-dependent).
- [ ] Branching uses `lax.cond` with consistent output structure.
- [ ] Static outputs in branches are wrapped (e.g., `eqxi.Static`).

---

## JIT / Static / PyTree Checklist

Use this before performance tuning or when encountering retracing.

### JIT boundaries

- [ ] JIT is applied once at the public API boundary.
- [ ] No nested JITs inside inner loops or solver steps.
- [ ] Large subgraphs are staged explicitly if compile time dominates.

### Static vs dynamic

- [ ] All non-array config is static (fields or partitioned).
- [ ] No Python objects or containers flow as dynamic inputs.
- [ ] `eqx.filter_closure_convert` is used for functions passed across JIT/AD boundaries.

### PyTree stability

- [ ] State treedef and leaf shapes/dtypes are unchanged across iterations.
- [ ] `jax.eval_shape`/`ShapeDtypeStruct` is used to validate structure.
- [ ] `eqx.partition` + `eqx.combine` are used to keep static leaves fixed.

### Control flow

- [ ] No Python `if/for/while` branches depend on arrays inside JIT.
- [ ] `lax.cond` branches return identical PyTree structure.
- [ ] `lax.scan` carries only fixed-shape data.

---

## Numerics / AD / Testing Checklist

Use this for solver implementation and verification.

### Dtype and stability

- [ ] Inputs are cast once to an inexact dtype at the boundary.
- [ ] Dtype policy (x32/x64) is explicit and documented.
- [ ] Divisions and norms are guarded against zero/inf.
- [ ] Nonfinite values are detected early and surfaced predictably.

### PRNG discipline

- [ ] PRNGKeys are explicit inputs/outputs to stochastic functions.
- [ ] Keys are split/folded deterministically (step/time included).
- [ ] PyTree-shaped randomness uses split-by-tree helpers.

### Custom AD

- [ ] Custom JVP/VJP is used for implicit/iterative methods where needed.
- [ ] Custom primitives include abstract_eval + JVP + transpose.
- [ ] Nondifferentiable config/state is guarded (`eqx.nondifferentiable`).

### Testing

- [ ] JIT + vmap + grad are exercised in tests.
- [ ] JVPs are checked against finite differences where feasible.
- [ ] Batching invariants hold (vmapped vs unbatched consistency).
- [ ] Failure modes (max_steps, nonfinite, ill-conditioned) are tested.

### Debugging

- [ ] `eqx.error_if` is used for runtime errors inside JIT.
- [ ] `jax.debug.print` is used for traced diagnostics.
- [ ] `EQX_ON_ERROR=breakpoint` is documented for runtime failures.

---

## Linear Algebra Checklist

Use this for linear solves, least squares, Jacobian operators, and preconditioning.

### Operator vs matrix

- [ ] Use operator-based representations when matrices are large or implicit.
- [ ] Materialize matrices only when necessary and justified by performance.
- [ ] Keep operator tags accurate (e.g., symmetric/PSD); incorrect tags are unsafe.

### Shape/structure

- [ ] Vector and operator structures match (PyTree shape/dtype compatible).
- [ ] Use `jax.eval_shape`/`ShapeDtypeStruct` to validate structure early.
- [ ] Keep operator input/output structures static.

### Solver configuration

- [ ] Choose solver based on conditioning and matrix structure.
- [ ] Handle under/overdetermined systems explicitly (least-squares/min-norm).
- [ ] Expose tolerances (`rtol`, `atol`) and max-iteration controls.

### Numerical stability

- [ ] Guard ill-conditioned solves with scaling or regularization.
- [ ] Prefer stable norm/conditioning checks over raw inverse operations.
- [ ] Detect nonfinite outputs and propagate result codes.

### AD considerations

- [ ] Use custom JVP/VJP when default gradients are unstable.
- [ ] Mark solver state/options as nondifferentiable.
- [ ] Ensure transpose/adjoint operations are implemented where required.

### Testing

- [ ] Compare against known solutions for small systems.
- [ ] Test singular/ill-conditioned cases and failure paths.
- [ ] Validate JVPs for linear solves when AD is supported.

---

## Testing Checklist (JAX/Equinox Numerics)

Use this to validate correctness, AD, and batching behavior.

### Coverage targets

- [ ] Parametrize over solvers, options, dtypes, and representative problems.
- [ ] Use deterministic PRNG fixtures (no global keys).
- [ ] Run core tests under JIT (`eqx.filter_jit` / `jax.jit`).
- [ ] Run core tests under vmap (`eqx.filter_vmap` / `jax.vmap`).

### Correctness

- [ ] Compare against known solutions for small problems.
- [ ] Validate result codes with `throw=False` where applicable.
- [ ] Include singular/ill-conditioned and nonfinite cases.

### AD checks

- [ ] Compare JVPs to finite differences for key APIs.
- [ ] Test reverse-mode gradients when supported.
- [ ] Confirm AD behavior under JIT and vmap.

### PyTree + static/dynamic behavior

- [ ] Partition dynamic/static args in tests and recombine inside the function.
- [ ] Confirm PyTree structure stability across iterations.

### Performance regressions (smoke)

- [ ] vmapped vs unbatched consistency.
- [ ] Basic compile/run time sanity for representative shapes.

---

## Project Engineering Checklist

Use this for general software engineering hygiene around numerical/JAX projects.

### API stability

- [ ] Public API signatures and return structures are stable and documented.
- [ ] Any breaking change includes a deprecation period or migration guide.
- [ ] Error semantics are consistent (`throw=False` paths tested).

### Documentation

- [ ] Public APIs and Modules have docstrings.
- [ ] Docstring format is consistent with repo style (e.g., `**Arguments:**` / `**Returns:**`).
- [ ] Examples cover common usage patterns and failure modes.

### CI / Quality gates

- [ ] Format/lint/typecheck/tests run in CI.
- [ ] New features include tests (including JIT/AD/vmap where relevant).
- [ ] Bug fixes include regression tests.

### Reproducibility

- [ ] PRNGKeys are explicit and deterministic in tests/examples.
- [ ] Device/backend nondeterminism is documented if it affects results.

### Serialization / checkpoints

- [ ] Persisted state is versioned.
- [ ] Load paths validate version compatibility or provide migration notes.

### Logging / diagnostics

- [ ] Diagnostic output is gated behind flags.
- [ ] No `print` in traced/JIT paths; use `jax.debug.print` sparingly.

---

# Code Snippets

Ready-to-use examples aligned with the JAX/Equinox guidance.

## Abstract Module Pattern

```python
import abc
import equinox as eqx
from typing import Generic, TypeVar
from jaxtyping import Array, PyTree


_State = TypeVar("_State")


class AbstractSolver(eqx.Module, Generic[_State]):
    rtol: eqx.AbstractVar[float]

    @abc.abstractmethod
    def init(self, y: PyTree[Array]) -> _State: ...

    @abc.abstractmethod
    def step(self, y: PyTree[Array], state: _State) -> tuple[PyTree[Array], _State]: ...


class MySolver(AbstractSolver[dict]):
    rtol: float

    def init(self, y):
        return {"count": 0}

    def step(self, y, state):
        return y, {"count": state["count"] + 1}
```

## JIT Boundary

```python
import equinox as eqx
import jax.numpy as jnp


class Config(eqx.Module):
    rtol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)


@eqx.filter_jit
def solve(fn, y0, args, cfg: Config):
    """Public JIT boundary: static config, dynamic arrays."""
    return fn(y0, args)


def fn(y, args):
    return y + args


y0 = jnp.ones((3,))
args = jnp.ones((3,))
_cfg = Config(rtol=1e-6, max_steps=128)
solve(fn, y0, args, _cfg)
```

## Partition Static State

```python
import equinox as eqx
import jax.numpy as jnp


class State(eqx.Module):
    buffer: jnp.ndarray
    meta: int = eqx.field(static=True)


state = State(buffer=jnp.zeros((4,)), meta=3)
state_dyn, state_static = eqx.partition(state, eqx.is_array)

# Update dynamic only
new_buffer = state_dyn.buffer + 1.0
new_state = State(buffer=new_buffer, meta=state_static.meta)

new_dyn, new_static = eqx.partition(new_state, eqx.is_array)
assert eqx.tree_equal(state_static, new_static) is True
state = eqx.combine(new_static, new_dyn)
```

## Filter Vmap Batching

```python
import equinox as eqx
import jax.numpy as jnp


@eqx.filter_jit
def solve(x, scale: float):
    return x * scale


batched = eqx.filter_vmap(solve, in_axes=(0, None))
xs = jnp.ones((8, 3))
ys = batched(xs, 2.0)
```

## PRNG Split by Tree

```python
import jax
import jax.random as jr
import jax.tree_util as jtu
import jax.numpy as jnp


def split_by_tree(key, tree):
    treedef = jtu.tree_structure(tree)
    return jtu.tree_unflatten(treedef, jr.split(key, treedef.num_leaves))


shape_tree = {"a": jax.ShapeDtypeStruct((2, 3), jnp.float32), "b": jax.ShapeDtypeStruct((4,), jnp.float32)}
key = jr.PRNGKey(0)
keys = split_by_tree(key, shape_tree)

samples = jtu.tree_map(lambda k, s: jr.normal(k, s.shape, s.dtype), keys, shape_tree)
```

## Custom JVP for Stable Norm

```python
import jax
import jax.numpy as jnp


@jax.custom_jvp
def stable_norm(x):
    return jnp.sqrt(jnp.sum(x * x))


@stable_norm.defjvp
def _stable_norm_jvp(primals, tangents):
    (x,), (tx,) = primals, tangents
    y = stable_norm(x)
    denom = jnp.where(y == 0, 1.0, y)
    return y, jnp.where(y == 0, 0.0, jnp.vdot(x, tx) / denom)
```

## Filter Cond with Static Outputs

```python
import equinox as eqx
import equinox.internal as eqxi
import jax.lax as lax


def filter_cond(pred, true_fun, false_fun, *operands):
    dyn, stat = eqx.partition(operands, eqx.is_array)

    def _wrap(fn):
        def inner(dyn_ops):
            out = fn(*eqx.combine(dyn_ops, stat))
            dyn_out, stat_out = eqx.partition(out, eqx.is_array)
            return dyn_out, eqxi.Static(stat_out)
        return inner

    dyn_out, stat_out = lax.cond(pred, _wrap(true_fun), _wrap(false_fun), dyn)
    return eqx.combine(dyn_out, stat_out.value)
```

## Linear Operator Pattern

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx


def f(x, args):
    return jnp.sin(x) + args


x = jnp.ones((4,))
args = jnp.ones((4,))

# Closure-convert to avoid closed-over tracers
fn = eqx.filter_closure_convert(f, x, args)

# Jacobian as a linear operator; no explicit matrix materialization
op = lx.JacobianLinearOperator(fn, x, args)

# Apply operator (JVP) and its transpose (VJP)
v = jnp.ones_like(x)
Jv = op.mv(v)
JT = op.transpose().mv(v)
```

## Implicit JVP Pattern

```python
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx


# Global functions only: no closed-over JAX arrays.

def fn_primal(inputs):
    y, a = inputs
    # Example: solve y^2 = a for y (pick positive branch)
    root = jnp.sqrt(a)
    residual = root * root - a
    return root, residual


def fn_rewrite(root, residual, inputs):
    # Rewrite whose Jacobian w.r.t. root is used by implicit JVP
    y, a = inputs
    return root * root - a


inputs = (jnp.array(1.0), jnp.array(4.0))
root, residual = optx.implicit_jvp(
    fn_primal,
    fn_rewrite,
    inputs,
    tags=frozenset(),
    linear_solver=lx.AutoLinearSolver(well_posed=True),
)
```

## Test JVP vs Finite Difference

```python
import functools as ft

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu


def finite_difference_jvp(fn, primals, tangents, eps=1e-5, **kwargs):
    primals_plus = jtu.tree_map(lambda x, t: x + eps * t, primals, tangents)
    primals_minus = jtu.tree_map(lambda x, t: x - eps * t, primals, tangents)
    f_plus = fn(*primals_plus, **kwargs)
    f_minus = fn(*primals_minus, **kwargs)
    return jtu.tree_map(lambda a, b: (a - b) / (2 * eps), f_plus, f_minus)


def getkey(seed=0):
    key = jr.PRNGKey(seed)
    while True:
        key, subkey = jr.split(key)
        yield subkey


def test_jvp_matches_finite_difference():
    keygen = getkey(0)
    key = next(keygen)

    def f(x, y):
        return jnp.sin(x) + y

    x = jr.normal(key, (4,))
    y = jr.normal(next(keygen), (4,))
    tx = jr.normal(next(keygen), (4,))
    ty = jr.normal(next(keygen), (4,))

    def wrapped(x, y):
        return f(x, y)

    # JVP under JIT
    out, t_out = eqx.filter_jit(ft.partial(eqx.filter_jvp, wrapped))(
        (x, y), (tx, ty)
    )

    # Finite difference baseline
    t_expected = finite_difference_jvp(wrapped, (x, y), (tx, ty), eps=1e-5)

    assert jtu.tree_all(
        jtu.tree_map(lambda a, b: jnp.allclose(a, b, atol=1e-4, rtol=1e-4), t_out, t_expected)
    )
```
