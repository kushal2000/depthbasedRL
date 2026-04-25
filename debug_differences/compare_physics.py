"""Compare per-actor physics params between Isaac Gym and Isaac Sim.

Reads the JSON dumps produced by ``physics_dump_isaacgym.py`` and
``physics_dump_isaacsim.py`` and prints a side-by-side comparison for:

  - sim cfg (dt, substeps / decimation, solver iters, physx defaults)
  - per-actor (robot/table/object) link mass, inertia summary, COM
  - per-actor shape contact_offset, rest_offset, friction, restitution

The two backends expose physics through different APIs (Isaac Gym:
``gym.get_actor_*_properties`` returning per-shape friction/restitution;
Isaac Sim: USD attributes + bound PhysicsMaterial), so the dumps don't
share an exact 1:1 schema. This script normalizes via summary stats
(min / max / median across links/shapes) for the comparable fields.

    python debug_differences/compare_physics.py

If the two JSONs aren't on disk yet, prompts you to run the dumpers in
their respective venvs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(p: Path) -> dict:
    if not p.exists():
        raise SystemExit(
            f"Missing: {p}\n"
            "Run:\n"
            "  .venv/bin/python debug_differences/physics_dump_isaacgym.py\n"
            "  .venv_isaacsim/bin/python debug_differences/physics_dump_isaacsim.py"
        )
    return json.loads(p.read_text())


def _stats(values: list) -> dict:
    """Numeric summary; ignores None entries."""
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return {"count": 0, "min": None, "max": None, "median": None}
    return {
        "count": len(nums),
        "min": min(nums),
        "max": max(nums),
        "median": median(nums),
    }


def _row(label: str, gym_val, sim_val, fmt: str = "{}") -> str:
    def f(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return fmt.format(v)
        return str(v)
    return f"  {label:<38} gym={f(gym_val):<22} sim={f(sim_val):<22}"


def _compare_sim_cfg(g: dict, s: dict) -> None:
    print("\n=== Sim / PhysX config ===")
    g_cfg = g.get("sim_cfg", {})
    s_cfg = s.get("sim_cfg", {})

    print(_row("dt (physics)", g_cfg.get("dt"), s_cfg.get("dt"), "{:.6f}"))
    g_substeps = g_cfg.get("substeps")
    s_decim = s_cfg.get("decimation")
    if g_substeps is not None:
        g_effective = g_cfg.get("dt", 0.0) * (g_substeps or 1)
    else:
        g_effective = None
    if s_decim is not None:
        s_effective = s_cfg.get("dt", 0.0) * (s_decim or 1)
    else:
        s_effective = None
    print(_row("substeps (gym) / decimation (sim)", g_substeps, s_decim))
    print(_row("effective policy step (s)", g_effective, s_effective, "{:.6f}"))
    print(_row("solver pos iters", g_cfg.get("physx_solver_position_iterations"),
               s_cfg.get("physx_solver_position_iterations_default")))
    print(_row("solver vel iters", g_cfg.get("physx_solver_velocity_iterations"),
               s_cfg.get("physx_solver_velocity_iterations_default")))
    print(_row("contact_offset default (cfg)",
               g_cfg.get("physx_contact_offset_default"), None, "{:.4f}"))
    print(_row("rest_offset default (cfg)",
               g_cfg.get("physx_rest_offset_default"), None, "{:.4f}"))
    print(_row("bounce threshold velocity",
               g_cfg.get("physx_bounce_threshold_velocity"),
               s_cfg.get("physx_bounce_threshold_velocity"), "{:.4f}"))
    print(_row("friction offset threshold",
               g_cfg.get("physx_friction_offset_threshold"),
               s_cfg.get("physx_friction_offset_threshold"), "{:.4f}"))


def _compare_role(g: dict, s: dict, role: str) -> None:
    g_role = g.get(role, {})
    s_role = s.get(role, {})
    print(f"\n=== {role.upper()} ===")
    g_links = g_role.get("links", [])
    s_links = s_role.get("links", [])
    g_shapes = g_role.get("shapes", [])
    s_shapes = s_role.get("shapes", [])

    print(_row("# rigid bodies / links", len(g_links), len(s_links)))
    print(_row("# collision shapes", len(g_shapes), len(s_shapes)))

    g_mass = _stats([l.get("mass") for l in g_links])
    s_mass = _stats([l.get("mass") for l in s_links])
    print(_row("mass total (kg, sum)",
               sum(l.get("mass", 0) or 0 for l in g_links),
               sum(l.get("mass", 0) or 0 for l in s_links), "{:.4f}"))
    print(_row("mass min / max",
               (g_mass["min"], g_mass["max"]),
               (s_mass["min"], s_mass["max"])))

    # Inertia diagonal magnitude (sum of three diag entries) per link
    g_ix = []
    for l in g_links:
        di = l.get("inertia_diag")
        if di:
            g_ix.append(sum(di))
    s_ix = []
    for l in s_links:
        di = l.get("inertia_diag")
        if di:
            s_ix.append(sum(di))
    g_in = _stats(g_ix)
    s_in = _stats(s_ix)
    print(_row("Σ(inertia_diag) median", g_in["median"], s_in["median"], "{:.6f}"))

    # Friction:
    #   gym: per-shape `friction` field
    #   sim: per-shape bound material -> static_friction
    g_fric = _stats([sp.get("friction") for sp in g_shapes])
    s_fric_static = _stats([
        (sp.get("material") or {}).get("static_friction")
        for sp in s_shapes
    ])
    s_fric_dyn = _stats([
        (sp.get("material") or {}).get("dynamic_friction")
        for sp in s_shapes
    ])
    print(_row("friction count authored",
               g_fric["count"], s_fric_static["count"]))
    print(_row("friction min / max (gym)",
               (g_fric["min"], g_fric["max"]), None))
    print(_row("static_friction min / max (sim)",
               None, (s_fric_static["min"], s_fric_static["max"])))
    print(_row("dynamic_friction min / max (sim)",
               None, (s_fric_dyn["min"], s_fric_dyn["max"])))

    g_rest = _stats([sp.get("restitution") for sp in g_shapes])
    s_rest = _stats([
        (sp.get("material") or {}).get("restitution") for sp in s_shapes
    ])
    print(_row("restitution median",
               g_rest["median"], s_rest["median"], "{:.4f}"))

    g_co = _stats([sp.get("contact_offset") for sp in g_shapes])
    s_co = _stats([sp.get("contact_offset") for sp in s_shapes])
    print(_row("contact_offset count authored",
               g_co["count"], s_co["count"]))
    print(_row("contact_offset min / max",
               (g_co["min"], g_co["max"]), (s_co["min"], s_co["max"])))

    g_ro = _stats([sp.get("rest_offset") for sp in g_shapes])
    s_ro = _stats([sp.get("rest_offset") for sp in s_shapes])
    print(_row("rest_offset count authored",
               g_ro["count"], s_ro["count"]))
    print(_row("rest_offset min / max",
               (g_ro["min"], g_ro["max"]), (s_ro["min"], s_ro["max"])))

    s_ce = [sp.get("collision_enabled") for sp in s_shapes]
    s_ce_n_true = sum(1 for v in s_ce if v is True)
    s_ce_n_false = sum(1 for v in s_ce if v is False)
    s_ce_n_default = sum(1 for v in s_ce if v is None)
    print(_row("collision_enabled (sim shapes T/F/default)",
               None, f"{s_ce_n_true}/{s_ce_n_false}/{s_ce_n_default}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym_json", type=str,
                        default=str(REPO_ROOT / "debug_differences/data/physics/isaacgym.json"))
    parser.add_argument("--sim_json", type=str,
                        default=str(REPO_ROOT / "debug_differences/data/physics/isaacsim.json"))
    args = parser.parse_args()

    g = _load(Path(args.gym_json))
    s = _load(Path(args.sim_json))

    print(f"loaded gym dump: {args.gym_json}")
    print(f"loaded sim dump: {args.sim_json}")

    _compare_sim_cfg(g, s)
    for role in ("robot", "table", "object"):
        _compare_role(g, s, role)

    if "goal_viz" in s:
        print("\n=== GOAL_VIZ (sim only) ===")
        gv = s["goal_viz"]
        gv_links = gv.get("links", [])
        gv_shapes = gv.get("shapes", [])
        print(f"  # links: {len(gv_links)}, # shapes: {len(gv_shapes)}")
        ce_t = sum(1 for sp in gv_shapes if sp.get("collision_enabled") is True)
        ce_f = sum(1 for sp in gv_shapes if sp.get("collision_enabled") is False)
        ce_d = sum(1 for sp in gv_shapes if sp.get("collision_enabled") is None)
        print(f"  collision_enabled (T/F/default): {ce_t}/{ce_f}/{ce_d}")
        co = _stats([sp.get("contact_offset") for sp in gv_shapes])
        print(f"  contact_offset min/max: {co['min']} / {co['max']}")


if __name__ == "__main__":
    main()
