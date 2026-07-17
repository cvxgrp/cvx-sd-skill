"""Canonical masked signal-decomposition problem builder (CVXPY).

This module is the keystone of the skill. It builds the convex signal
decomposition (SD) problem

    minimize    phi_1(x1) + phi_2(x2) + ... + phi_K(xK)
    subject to  y == x1 + x2 + ... + xK   (over observed entries only)

following the Meyers & Boyd framework, with two invariants enforced *by
construction*:

1. **x1 is always the residual** (mean-square-small, or a robust variant).
   Structural components are x2, x3, ... and are appended in order, so
   extending a model never renumbers anything.
2. **Missing data is native.** The linking (consistency) equality is imposed
   only on observed entries via a boolean mask; unobserved entries (NaN in
   ``y``) are simply not constrained. Held-out validation data is the *same*
   mechanism -- mask a known entry and score the imputed value against truth.

Components are represented as plain callables (see :class:`Component`) that,
given the series length ``T``, return their CVXPY variable/expression, loss,
and constraints. The richer catalog of convex component builders lives in
``components.py``; this module only defines the residual and the assembly.
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
    """

    role: str
    build: Callable[[int], tuple[cp.Expression, cp.Expression, list]]
    aux: dict[str, cp.Expression] = field(default_factory=dict)


def _resolve_residual_loss(residual_loss):
    """Resolve a residual-loss specifier to a callable ``x1 -> scalar``.

    Accepts either a callable (returned unchanged) or one of the string preset
    names, which map to the factories in :mod:`data_fidelity`. Passing a
    callable is the general case; the strings are convenience aliases for the
    most common losses.

    Parameters
    ----------
    residual_loss : callable or str
        A convex ``loss_fn(x1) -> scalar cvxpy expression``, or one of
        ``"l2"``, ``"l1"``, ``"huber"``, ``"quantile"`` (using default
        parameters). For non-default preset parameters (e.g. a specific Huber
        threshold or quantile level), pass the factory result directly, e.g.
        ``residual_loss=huber_loss(M=0.5)``.

    Returns
    -------
    callable
        ``loss_fn(x1) -> scalar cvxpy expression``.
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

    Assembles ``y = x1 + x2 + ... + xK`` where ``x1`` is the residual
    (index 1, always) and ``components`` supply the structural terms
    ``x2, ..., xK`` in order. The consistency equality is imposed only on
    the observed (non-NaN) entries of ``y``.

    Parameters
    ----------
    y : numpy.ndarray, shape (T,)
        Observed scalar signal. ``NaN`` entries are treated as missing and
        excluded from the linking constraint.
    components : list of Component
        Structural components (x2, ..., xK), in order.
    residual_loss : callable or str
        Convex loss for the residual x1. Any DCP-compliant
        ``loss_fn(x1) -> scalar cvxpy expression`` is accepted; this is the
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
          ``"residual"`` for x1, plus any component ``aux`` expressions.
        - ``"residual"`` : the residual variable x1 (also under
          ``variables["residual"]``).
        - ``"mask"`` : boolean array of observed entries.
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
    mask = ~np.isnan(y)
    if not mask.any():
        raise ValueError("y has no observed (non-NaN) entries.")

    # x1: the residual, always index 1.
    x1 = cp.Variable(T, name="residual")
    loss_fn = _resolve_residual_loss(residual_loss)
    objective = loss_fn(x1)
    total = x1

    variables: dict[str, cp.Expression] = {"residual": x1}
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
        for aux_name, aux_expr in comp.aux.items():
            variables[aux_name] = aux_expr

    # Masked linking / consistency equality: observed entries only.
    constraints.append(y[mask] == total[mask])

    problem = cp.Problem(cp.Minimize(objective), constraints)
    return {
        "problem": problem,
        "variables": variables,
        "residual": x1,
        "mask": mask,
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
