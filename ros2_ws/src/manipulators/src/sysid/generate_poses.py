#!/usr/bin/env python3
"""Generate collision-free joint configurations for gravity sysid.

Three stages:
  1. Static validation — LHS candidates checked for collision at the target
     pose (fast, no motion sim). Produces green/red markers in viewer.
  2. Nearest-neighbor ordering — greedy TSP to minimise joint travel.
  3. Transition validation — simulate motion between consecutive poses,
     drop pairs that collide in transit. Draws path lines in viewer.

No ROS required.

Usage:
    python generate_poses.py --n-candidates 500 --output poses.yaml
"""

import os
import argparse
import numpy as np
import yaml
import mujoco
import mujoco.viewer

DEFAULT_XML = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "assets", "robots", "kinova", "mjcf", "gen3.xml",
)

# Sampling limits (degrees). Each joint has a list of (min, max) intervals.
# Use multiple intervals for mirrored ranges, e.g., [(-90, -60), (60, 90)]
# Continuous joints (1,3,5,7): ±143 deg (~2.5 rad)
# Revolute joints: slightly inside gen3.xml ctrlrange
JOINT_RANGES_DEG = [
    [(-143, 143)],    # J1 continuous
    [(-120, -60), (60,120)],    # J2 (ctrlrange ±2.25 rad ≈ ±129°)
    [(-143, 143)],    # J3 continuous
    [(-30, 30), (-30,30)],    # J4 (ctrlrange ±2.58 rad ≈ ±148°)
    [(-143, 143)],    # J5 continuous
    [(-30, 30), (-30,30)],    # J6 (ctrlrange ±2.10 rad ≈ ±120°)
    [(-143, 143)],    # J7 continuous
]

SCENE_TEMPLATE = """\
<mujoco model="sysid_scene">
  <include file="{gen3_file}"/>
  <visual>
    <global offwidth="800" offheight="600"/>
  </visual>
  <worldbody>
{lighting}
    <body name="table">
      <geom name="table_geom" type="box" pos="0 0 0.125"
            size="1.6 1.6 0.025" rgba="0.6 0.5 0.4 1"/>
    </body>
{markers}
  </worldbody>
  <contact>
    <exclude body1="base_link" body2="table"/>
    <exclude body1="base_link" body2="shoulder_link"/>
    <exclude body1="base_mount" body2="table"/>
    <exclude body1="base" body2="table"/>
  </contact>
</mujoco>
"""


def latin_hypercube_with_ranges(n, joint_ranges_deg, rng):
    """Latin hypercube sampling with custom joint ranges (in degrees).

    joint_ranges_deg: list of 7 elements, each is a list of (lo, hi) tuples in degrees.
                      For mirrored ranges, use multiple tuples, e.g., [(-90, -60), (60, 90)].
    Returns: configs in radians.
    """
    configs = np.zeros((n, 7))

    for j in range(7):
        ranges = joint_ranges_deg[j]

        if len(ranges) == 1:
            # Simple single range
            lo, hi = ranges[0]
            edges = np.linspace(lo, hi, n + 1)
            for i in range(n):
                configs[i, j] = rng.uniform(edges[i], edges[i + 1])
            rng.shuffle(configs[:, j])
        else:
            # Multiple ranges (e.g., mirrored)
            # Calculate total span and allocate samples proportionally
            spans = [(hi - lo) for lo, hi in ranges]
            total_span = sum(spans)

            # Allocate samples to each range proportionally
            allocations = []
            remaining = n
            for k, span in enumerate(spans):
                if k == len(spans) - 1:
                    alloc = remaining  # Last range gets the rest
                else:
                    alloc = int(round(n * span / total_span))
                    remaining -= alloc
                allocations.append(alloc)

            # Generate LHS samples for each sub-range
            values = []
            for (lo, hi), alloc in zip(ranges, allocations):
                if alloc > 0:
                    edges = np.linspace(lo, hi, alloc + 1)
                    for i in range(alloc):
                        values.append(rng.uniform(edges[i], edges[i + 1]))

            values = np.array(values)
            rng.shuffle(values)
            configs[:, j] = values

    # Convert to radians
    return np.deg2rad(configs)


def load_scene(gen3_xml_path, n_green=0, n_red=0, light=False):
    """Load gen3 model + table + optional mocap marker spheres.
    Green markers first (indices 0..n_green-1), then red (n_green..n_green+n_red-1)."""
    mjcf_dir = os.path.dirname(os.path.abspath(gen3_xml_path))
    marker_lines = []
    for i in range(n_green):
        marker_lines.append(
            f'    <body name="mg_{i}" mocap="true" pos="0 0 -1">'
            f'<geom type="sphere" size="0.012" rgba="0 1 0 0.7" '
            f'contype="0" conaffinity="0"/></body>'
        )
    for i in range(n_red):
        marker_lines.append(
            f'    <body name="mr_{i}" mocap="true" pos="0 0 -1">'
            f'<geom type="sphere" size="0.008" rgba="1 0 0 0.5" '
            f'contype="0" conaffinity="0"/></body>'
        )
    if light:
        lighting_block = (
            '    <light name="top" pos="0 0 2" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>\n'
            '    <light name="front" pos="1 -1 1.5" dir="-1 1 -1" diffuse="0.4 0.4 0.4"/>\n'
            '    <geom name="floor" type="plane" size="2 2 0.01" pos="0 0 -0.05"\n'
            '          rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>'
        )
    else:
        lighting_block = ''
    scene_xml = SCENE_TEMPLATE.format(
        gen3_file=os.path.basename(gen3_xml_path),
        lighting=lighting_block,
        markers="\n".join(marker_lines),
    )
    scene_path = os.path.join(mjcf_dir, "_sysid_tmp.xml")
    try:
        with open(scene_path, "w") as f:
            f.write(scene_xml)
        model = mujoco.MjModel.from_xml_path(scene_path)
    finally:
        if os.path.exists(scene_path):
            os.remove(scene_path)
    return model


def set_arm_state(model, data, q_rad):
    """Set the arm to a specific joint configuration (position, zero velocity)."""
    data.qpos[:7] = q_rad
    data.qvel[:7] = 0.0
    mujoco.mj_forward(model, data)


def get_geom_label(model, geom_id):
    """Get a readable label for a geom (prefer body name if geom unnamed)."""
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if geom_name:
        return geom_name
    # Geom unnamed, use parent body name
    body_id = model.geom_bodyid[geom_id]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return body_name or f"body{body_id}"


def get_collision_info(model, data):
    """Get info about current collisions."""
    collisions = []
    for i in range(data.ncon):
        contact = data.contact[i]
        label1 = get_geom_label(model, contact.geom1)
        label2 = get_geom_label(model, contact.geom2)
        collisions.append((label1, label2))
    return collisions


def simulate_transition(model, data, start_rad, target_rad,
                        max_steps=3000, tol=0.03, viewer=None,
                        n_waypoints=20, steps_per_waypoint=150):
    """Simulate start → target with interpolated waypoints to avoid overshoot.

    Returns: (success, info_dict)
        success: True if no collision and converged
        info_dict: {'collisions': [...], 'q_deg': [...], 'step': int} or None
    """
    saved_mocap = data.mocap_pos.copy() if model.nmocap > 0 else None
    data.time = 0.0
    set_arm_state(model, data, start_rad)
    if saved_mocap is not None:
        data.mocap_pos[:] = saved_mocap

    # Interpolate waypoints from start to target
    alphas = np.linspace(0, 1, n_waypoints + 1)[1:]  # skip 0 (start)

    global_step = 0
    for waypoint_idx, alpha in enumerate(alphas):
        waypoint = start_rad + alpha * (target_rad - start_rad)
        data.ctrl[:7] = waypoint

        for step in range(steps_per_waypoint):
            mujoco.mj_step(model, data)
            global_step += 1
            if viewer is not None:
                viewer.sync()
            if data.ncon > 0:
                info = {
                    'collisions': get_collision_info(model, data),
                    'q_deg': np.rad2deg(data.qpos[:7]).tolist(),
                    'step': global_step,
                }
                return False, info

            # Check if reached this waypoint
            if (np.max(np.abs(data.qpos[:7] - waypoint)) < tol
                    and np.max(np.abs(data.qvel[:7])) < 0.15):
                break

    # Final convergence check
    pos_err = np.max(np.abs(data.qpos[:7] - target_rad))
    vel_err = np.max(np.abs(data.qvel[:7]))
    converged = pos_err < tol and vel_err < 0.15
    if not converged:
        info = {
            'collisions': [],
            'q_deg': np.rad2deg(data.qpos[:7]).tolist(),
            'step': global_step,
            'timeout': True,
            'pos_err': pos_err,
            'vel_err': vel_err,
        }
        return False, info
    return True, None


def is_pose_collision_free(model, data, q_rad):
    """Check if a pose has any collision (static check, no motion sim)."""
    set_arm_state(model, data, q_rad)
    data.ctrl[:7] = q_rad
    # Need a few substeps for contact detection to settle
    for _ in range(5):
        mujoco.mj_step(model, data)
    return data.ncon == 0


def joint_distance(a, b):
    """Max absolute joint difference (Chebyshev distance)."""
    return np.max(np.abs(a - b))


def nearest_neighbor_order(poses, start=None):
    """Greedy nearest-neighbor ordering by joint distance. Returns index array."""
    n = len(poses)
    if start is None:
        start = np.zeros(poses.shape[1])
    visited = np.zeros(n, dtype=bool)
    order = []
    current = start
    for _ in range(n):
        dists = np.array([joint_distance(current, poses[j]) if not visited[j]
                          else np.inf for j in range(n)])
        idx = np.argmin(dists)
        visited[idx] = True
        order.append(idx)
        current = poses[idx]
    return np.array(order)


def nearest_neighbor_order_ee(ee_positions, start_ee):
    """Greedy nearest-neighbor ordering by EE Euclidean distance. Returns index array."""
    n = len(ee_positions)
    visited = np.zeros(n, dtype=bool)
    order = []
    current = start_ee
    for _ in range(n):
        dists = np.array([np.linalg.norm(current - ee_positions[j]) if not visited[j]
                          else np.inf for j in range(n)])
        idx = np.argmin(dists)
        visited[idx] = True
        order.append(idx)
        current = ee_positions[idx]
    return np.array(order)


def add_line(viewer, pos1, pos2, rgba):
    """Add a line segment to the viewer scene between two 3D positions."""
    if viewer is None:
        return
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_connector(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_LINE,
        3.0,  # width in pixels
        np.asarray(pos1, dtype=np.float64),
        np.asarray(pos2, dtype=np.float64),
    )
    scn.geoms[scn.ngeom].rgba[:] = rgba
    scn.ngeom += 1


def get_ee_pos(model, data, q_rad):
    """Forward kinematics to get end-effector position for a configuration."""
    saved_q = data.qpos[:7].copy()
    saved_v = data.qvel[:7].copy()
    data.qpos[:7] = q_rad
    data.qvel[:7] = 0.0
    mujoco.mj_forward(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector_link")
    pos = data.xpos[ee_id].copy()
    data.qpos[:7] = saved_q
    data.qvel[:7] = saved_v
    mujoco.mj_forward(model, data)
    return pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default=DEFAULT_XML, help="Path to gen3.xml")
    p.add_argument("--n-candidates", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="poses.yaml")
    p.add_argument("--visualize", action="store_true",
                   help="Open MuJoCo viewer to watch validation")
    p.add_argument("--light", action="store_true",
                   help="Add lights and floor to the scene")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.xml} ...")
    n_green = args.n_candidates if args.visualize else 0
    n_red = args.n_candidates if args.visualize else 0
    model = load_scene(args.xml, n_green=n_green, n_red=n_red, light=args.light)
    data = mujoco.MjData(model)

    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    "end_effector_link")

    # Generate candidates using JOINT_RANGES_DEG
    print("Joint ranges (deg):")
    for j, ranges in enumerate(JOINT_RANGES_DEG):
        range_strs = [f"[{lo}, {hi}]" for lo, hi in ranges]
        print(f"  J{j+1}: {' ∪ '.join(range_strs)}")
    candidates = latin_hypercube_with_ranges(args.n_candidates, JOINT_RANGES_DEG, rng)

    viewer = None
    if args.visualize:
        viewer = mujoco.viewer.launch_passive(model, data)

    GREEN = np.array([0, 255, 0, 180], dtype=np.uint8)

    # ── Stage 1: Static pose validation ──────────────────────────────
    print(f"[Stage 1] Checking {args.n_candidates} candidates for collision-free poses ...")
    valid = []
    n_green_placed = 0
    n_red_placed = 0

    for i, target in enumerate(candidates):
        if viewer is not None and not viewer.is_running():
            print("Viewer closed, stopping.")
            break

        if is_pose_collision_free(model, data, target):
            ee = data.xpos[ee_body_id].copy()
            if viewer is not None and n_green_placed < n_green:
                data.mocap_pos[n_green_placed] = ee
                viewer.sync()
                n_green_placed += 1
            valid.append(target)
            print(f"  [{len(valid):3d}] candidate {i+1:4d} OK  "
                  f"[{', '.join(f'{a:+.2f}' for a in target)}]")
        else:
            if viewer is not None and n_red_placed < n_red:
                ee = data.xpos[ee_body_id].copy()
                data.mocap_pos[n_green + n_red_placed] = ee
                viewer.sync()
                n_red_placed += 1

    print(f"  {len(valid)}/{args.n_candidates} collision-free "
          f"({len(valid)/args.n_candidates*100:.0f}% acceptance)")

    if len(valid) == 0:
        print("ERROR: no valid poses found.")
        return

    valid = np.array(valid)

    # ── Stage 2: Nearest-neighbor ordering ───────────────────────────
    print(f"\n[Stage 2] Ordering {len(valid)} poses by nearest-neighbor ...")
    order = nearest_neighbor_order(valid, start=np.zeros(7))
    valid = valid[order]
    total_dist = sum(joint_distance(valid[i], valid[i+1])
                     for i in range(len(valid)-1))
    print(f"  Total path distance (Chebyshev): {total_dist:.1f} rad")

    # ── Stage 3: Transition validation ───────────────────────────────
    print(f"\n[Stage 3] Validating motion transitions ...")

    # Move table down to z=5cm for motion validation
    table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
    if table_geom_id >= 0:
        model.geom_pos[table_geom_id][2] = 0.025  # table top at z=5cm
        print("  (table moved to z=5cm for motion validation)")

    keep = [True] * len(valid)
    via_zero = [False] * len(valid)  # Track which poses needed via-zero path
    current_pos = np.zeros(7)
    prev_ee = get_ee_pos(model, data, current_pos) if viewer else None

    zero_pose = np.zeros(7)
    zero_ee = get_ee_pos(model, data, zero_pose) if viewer else None

    try:
        for i, target in enumerate(valid):
            if viewer is not None and not viewer.is_running():
                print("Viewer closed, stopping.")
                break

            success, info = simulate_transition(model, data, current_pos, target,
                                                viewer=viewer)
            if success:
                ee = data.xpos[ee_body_id].copy()
                d = joint_distance(current_pos, target)
                if viewer is not None:
                    add_line(viewer, prev_ee, ee, GREEN)
                    viewer.sync()
                    prev_ee = ee
                current_pos = target.copy()
                print(f"  [{i+1:3d}/{len(valid)}] OK   dist={d:.2f}")
            else:
                # Direct path failed, try via zero pose
                if info and info['collisions']:
                    pairs = ", ".join(f"{g1}<->{g2}" for g1, g2 in info['collisions'][:3])
                    print(f"  [{i+1:3d}/{len(valid)}] direct FAIL: {pairs}")
                elif info and info.get('timeout'):
                    print(f"  [{i+1:3d}/{len(valid)}] direct FAIL: timeout (pos_err={info['pos_err']:.3f}, vel_err={info['vel_err']:.3f})")
                else:
                    print(f"  [{i+1:3d}/{len(valid)}] direct FAIL")

                # Go back to zero
                success_to_zero, _ = simulate_transition(model, data, current_pos, zero_pose,
                                                         viewer=viewer)
                if success_to_zero:
                    if viewer is not None:
                        add_line(viewer, prev_ee, zero_ee, GREEN)
                        viewer.sync()
                        prev_ee = zero_ee
                    current_pos = zero_pose.copy()

                    # Try zero → target
                    success_from_zero, info2 = simulate_transition(model, data, zero_pose, target,
                                                                   viewer=viewer)
                    if success_from_zero:
                        ee = data.xpos[ee_body_id].copy()
                        if viewer is not None:
                            add_line(viewer, prev_ee, ee, GREEN)
                            viewer.sync()
                            prev_ee = ee
                        current_pos = target.copy()
                        via_zero[i] = True  # Mark this pose as needing via-zero
                        print(f"        -> via zero: OK")
                    else:
                        keep[i] = False
                        if info2 and info2['collisions']:
                            pairs2 = ", ".join(f"{g1}<->{g2}" for g1, g2 in info2['collisions'][:3])
                            print(f"        -> via zero: SKIP ({pairs2})")
                        elif info2 and info2.get('timeout'):
                            print(f"        -> via zero: SKIP (timeout pos_err={info2['pos_err']:.3f})")
                        else:
                            print(f"        -> via zero: SKIP")
                else:
                    # Can't even go back to zero, skip this pose
                    keep[i] = False
                    print(f"        -> can't return to zero, SKIP")
    finally:
        if viewer is not None:
            viewer.close()

    final = valid[keep]
    final_via_zero = [v for v, k in zip(via_zero, keep) if k]
    n_dropped = len(valid) - len(final)
    n_via_zero = sum(final_via_zero)

    # Prepend zero pose as the first pose
    q_rad = np.vstack([np.zeros(7), final])
    q_deg = np.rad2deg(q_rad)

    # go_to_zero_before: False for first pose (already at zero), then the via_zero flags
    go_to_zero_before = [False] + final_via_zero

    output = {
        "n_poses": int(len(q_deg)),
        "n_via_zero": n_via_zero,
        "q_deg": q_deg.tolist(),
        "go_to_zero_before": go_to_zero_before,
    }
    with open(args.output, "w") as f:
        yaml.dump(output, f, default_flow_style=None, sort_keys=False)

    print(f"\n{len(q_rad)} final poses (incl. zero), {n_via_zero} need via-zero "
          f"({n_dropped} dropped). Saved to {args.output}")


if __name__ == "__main__":
    main()
