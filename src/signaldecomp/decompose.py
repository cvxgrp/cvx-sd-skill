"""Canonical masked signal-decomposition problem builder (CVXPY).

This module is the keystone of the skill. It builds the convex signal
decomposition (SD) problem

    minimize    phi_0(x0) + phi_1(x1) + ... + phi_K(xK)
    subject to  y == x0 + x1 + ... + xK   (over fitted entries only)

following the Meyers & Boyd framework, with two invariants enforced *by
construction*:

1. **x0 is always the residual** (mean-square-small, or a robust variant).
   Structural components are x1, x2, ... and are appended in order, so
   extending a model never renumbers anything.
2. **Unavailable data is native.** The linking (consistency) equality is
   imposed only where ``y`` is observed and every component input is valid.
   Missing observations, held-out rows, and unavailable exogenous offsets are
   excluded from the effective fitting mask.

Components are represented as plain callables (see :class:`Component`) that,
given the series length ``T``, return their CVXPY variable/expression, loss,
and constraints. Optional component masks declare input availability. The
richer catalog of convex component builders lives in ``components.py``; this
module only defines the residual and the assembly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cvxpy as cp
import numpy as np

from signaldecomp import data_fidelity

_OPTIMAL_STATUSES = ("optimal", "optimal_inaccurate")


@dataclass
class Component:
    """A structural signal-decomposition component.

    A component is defined by a ``build`` callable that, given the series
    length ``T``, returns the tuple ``(expr, loss, constraints)`` where

    - ``expr`` is the component signal as a CVXPY expression of length ``T``
      (typically a ``cp.Variable(T)`` directly, or e.g. ``B @ theta`` for a
      basis component),
    - ``loss`` is a scalar CVXPY expression (the convex penalty phi_k), and
    - ``constraints`` is a list of CVXPY constraints (possibly empty).

    Parameters
    ----------
    role : str
        Semantic name for the component (e.g. ``"trend"``, ``"seasonal"``).
        Downstream tools reference components by role, never by index.
    build : callable
        ``build(T) -> (expr, loss, constraints)``.
    aux : dict, optional
        Extra named CVXPY expressions to expose in the result (e.g. basis
        coefficients, trend slope), keyed by name.
    valid_mask : numpy.ndarray, optional
        Boolean vector marking rows where this component's inputs are
        available. ``make_problem`` intersects these masks with the observed
        signal mask before imposing consistency.
    parameterization_mask : numpy.ndarray, optional
        Exact fitting mask used for data-dependent preprocessing such as basis
        whitening or knot placement. When present, it must match the final
        linking mask exactly; a mismatch raises instead of silently leaking
        held-out rows or conditioning a different operator.
    metadata : dict, optional
        Non-CVXPY component metadata such as resolved knots or whitening
        diagnostics.
    """

    role: str
    build: Callable[[int], tuple[cp.Expression, cp.Expression, list]]
    aux: dict[str, cp.Expression] = field(default_factory=dict)
    valid_mask: np.ndarray | None = None
    parameterization_mask: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def _validate_component_mask(mask, T, *, role, name):
    """Validate a component mask without silently coercing numeric arrays."""
    arr = np.asarray(mask)
    if arr.dtype != np.bool_:
        raise TypeError(f"{name} for role {role!r} must have boolean dtype.")
    if arr.shape != (T,):
        raise ValueError(
            f"{name} for role {role!r} has shape {arr.shape}, expected ({T},)."
        )
    return arr


def _resolve_residual_loss(residual_loss):
    """Resolve a residual-loss specifier to a callable ``x -> scalar``.

    Accepts either a callable (returned unchanged) or one of the string preset
    names, which map to the factories in :mod:`data_fidelity`. Passing a
    callable is the general case; the strings are convenience aliases for the
    most common losses.

    Parameters
    ----------
    residual_loss : callable or str
        A convex ``loss_fn(x) -> scalar cvxpy expression``, or one of
        ``"l2"``, ``"l1"``, ``"huber"``, ``"quantile"`` (using default
        parameters). For non-default preset parameters (e.g. a specific Huber
        threshold or quantile level), pass the factory result directly, e.g.
        ``residual_loss=huber_loss(M=0.5)``.

    Returns
    -------
    callable
        ``loss_fn(x) -> scalar cvxpy expression``.
    """
    if callable(residual_loss):
        return residual_loss
    presets = {
        "l2": data_fidelity.l2_loss,
        "l1": data_fidelity.l1_loss,
        "huber": data_fidelity.huber_loss,
        "quantile": data_fidelity.quantile_loss,
    }
    if residual_loss in presets:
        return presets[residual_loss]()
    raise ValueError(
        f"residual_loss must be a callable or one of {sorted(presets)}; "
        f"got {residual_loss!r}"
    )


def make_problem(
    y: np.ndarray,
    components: list[Component],
    residual_loss="l2",
) -> dict:
    """Build the masked signal-decomposition problem.

    Assembles ``y = x0 + x1 + ... + xK`` where ``x0`` is the residual
    (index 0, always) and ``components`` supply the structural terms
    ``x1, ..., xK`` in order. The consistency equality is imposed on the
    intersection of observed ``y`` rows and component-valid rows.

    Parameters
    ----------
    y : numpy.ndarray, shape (T,)
        Observed scalar signal. Non-finite entries are treated as missing and
        excluded from the linking constraint.
    components : list of Component
        Structural components (x1, ..., xK), in order.
    residual_loss : callable or str
        Convex loss for the residual x0. Any DCP-compliant
        ``loss_fn(x) -> scalar cvxpy expression`` is accepted; this is the
        general, extensible case. The strings ``"l2"`` (default), ``"l1"``,
        ``"huber"``, ``"quantile"`` are convenience aliases for the presets in
        :mod:`data_fidelity` with default parameters. For non-default parameters,
        pass the factory result, e.g. ``residual_loss=huber_loss(M=0.5)``.

    Returns
    -------
    dict
        Keys:

        - ``"problem"`` : the :class:`cvxpy.Problem` (call ``.solve()`` first).
        - ``"variables"`` : dict mapping role -> CVXPY expression, including
          ``"residual"`` for x0, plus any component ``aux`` expressions.
        - ``"residual"`` : the residual variable x0 (also under
          ``variables["residual"]``).
        - ``"mask"`` : historical boolean array of observed ``y`` entries.
        - ``"component_mask"`` : boolean rows where every component input is
          available, independent of whether ``y`` is observed.
        - ``"fit_mask"`` : exact boolean rows used by the linking constraint.
        - ``"component_metadata"`` : per-role preprocessing/numerical metadata.
        - ``"args"`` : the scalar build arguments, for reproducible re-builds.

    Notes
    -----
    Roles must be unique; a duplicate role raises ``ValueError``. The residual
    role name ``"residual"`` is reserved.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"V1 supports scalar (1-D) signals only; got ndim={y.ndim}.")
    T = y.shape[0]
    observed_mask = np.isfinite(y)
    if not observed_mask.any():
        raise ValueError("y has no observed finite entries.")

    component_mask = np.ones(T, dtype=bool)
    for comp in components:
        if comp.valid_mask is not None:
            component_mask &= _validate_component_mask(
                comp.valid_mask, T, role=comp.role, name="valid_mask"
            )
    fit_mask = observed_mask & component_mask
    if not fit_mask.any():
        raise ValueError("no rows remain in the fitting mask after component validity.")

    for comp in components:
        if comp.parameterization_mask is not None:
            parameterization_mask = _validate_component_mask(
                comp.parameterization_mask,
                T,
                role=comp.role,
                name="parameterization_mask",
            )
            if not np.array_equal(parameterization_mask, fit_mask):
                raise ValueError(
                    f"parameterization_mask for role {comp.role!r} must exactly "
                    "match the final fit_mask used by the linking constraint."
                )

    # x0: the residual, always index 0.
    x0 = cp.Variable(T, name="residual")
    loss_fn = _resolve_residual_loss(residual_loss)
    objective = loss_fn(x0)
    total = x0

    variables: dict[str, cp.Expression] = {"residual": x0}
    component_metadata: dict[str, dict[str, object]] = {}
    constraints: list = []
    seen_roles = {"residual"}

    for comp in components:
        if comp.role in seen_roles:
            raise ValueError(f"duplicate component role {comp.role!r}")
        seen_roles.add(comp.role)
        expr, loss, cons = comp.build(T)
        objective = objective + loss
        constraints.extend(cons)
        total = total + expr
        variables[comp.role] = expr
        component_metadata[comp.role] = dict(comp.metadata)
        for aux_name, aux_expr in comp.aux.items():
            variables[aux_name] = aux_expr

    # Masked linking / consistency equality: observed entries only.
    constraints.append(y[fit_mask] == total[fit_mask])

    problem = cp.Problem(cp.Minimize(objective), constraints)
    return {
        "problem": problem,
        "variables": variables,
        "residual": x0,
        "mask": observed_mask,
        "component_mask": component_mask,
        "fit_mask": fit_mask,
        "component_metadata": component_metadata,
        "args": {"residual_loss": residual_loss},
    }


def solve(
    built: dict,
    solver: str = cp.CLARABEL,
    verify_dcp: bool = True,
    **solve_kwargs,
) -> dict:
    """Solve a built decomposition problem and return solved component values.

    Parameters
    ----------
    built : dict
        The return value of :func:`make_problem`.
    solver : str
        CVXPY solver to use. Defaults to CLARABEL. Always overridable.
    verify_dcp : bool
        If True (default), assert the problem is DCP before solving. DCP
        compliance is the "verifiable target" guarantee: a malformed convex
        model is caught here rather than producing a meaningless solution.
    **solve_kwargs
        Passed through to ``problem.solve``.

    Returns
    -------
    dict
        The ``built`` dict augmented with:

        - ``"status"`` : solver status string.
        - ``"values"`` : dict role -> solved numpy array (or scalar for aux
          scalars).

    Raises
    ------
    ValueError
        If the problem is not DCP (when ``verify_dcp``), or the solver does
        not reach an (inaccurate-)optimal status.
    """
    problem = built["problem"]
    if verify_dcp and not problem.is_dcp():
        raise ValueError("problem is not DCP; check component losses/constraints.")
    problem.solve(solver=solver, **solve_kwargs)
    if problem.status not in _OPTIMAL_STATUSES:
        raise ValueError(f"solver did not converge: status={problem.status!r}")
    values = {role: _solved_value(expr) for role, expr in built["variables"].items()}
    return {**built, "status": problem.status, "values": values}


def _solved_value(expr: cp.Expression):
    """Return a component's solved value as a plain float or numpy array.

    CVXPY returns a 0-d ndarray for scalar variables; collapse those to a
    Python float so scalar aux quantities (e.g. a trend slope) read naturally
    downstream. Vector components are returned as their numpy array. ``None``
    (unsolved) passes through unchanged.
    """
    value = expr.value
    if value is None:
        return None
    if np.ndim(value) == 0:
        return float(value)
    return value
