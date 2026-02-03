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

# Sampling limits (radians). Conservative to avoid cable wrap.
# Continuous joints (1,3,5,7): ±2.5 rad
# Revolute joints: slightly inside gen3.xml ctrlrange
JOINT_LIMITS_RAD = np.array([
    [-2.5,  2.5 ],  # J1 continuous
    [-2.20, 2.20],  # J2 (ctrlrange ±2.25)
    [-2.5,  2.5 ],  # J3 continuous
    [-2.50, 2.50],  # J4 (ctrlrange ±2.58)
    [-2.5,  2.5 ],  # J5 continuous
    [-2.05, 2.05],  # J6 (ctrlrange ±2.10)
    [-2.5,  2.5 ],  # J7 continuous
])

SCENE_TEMPLATE = """\
<mujoco model="sysid_scene">
  <include file="{gen3_file}"/>
  <visual>
    <global offwidth="800" offheight="600"/>
  </visual>
  <worldbody>
{lighting}
    <body name="table">
      <geom name="table_geom" type="box" pos="0 0 -0.025"
            size="0.8 0.8 0.025" rgba="0.6 0.5 0.4 1"/>
    </body>
{markers}
  </worldbody>
  <contact>
    <exclude body1="base_link" body2="table"/>
    <exclude body1="base_link" body2="shoulder_link"/>
  </contact>
</mujoco>
"""


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


def latin_hypercube(n, rng):
    """Latin hypercube sampling across joint limits (radians)."""
    configs = np.zeros((n, 7))
    for j in range(7):
        lo, hi = JOINT_LIMITS_RAD[j]
        edges = np.linspace(lo, hi, n + 1)
        for i in range(n):
            configs[i, j] = rng.uniform(edges[i], edges[i + 1])
        rng.shuffle(configs[:, j])
    return configs


def set_arm_state(model, data, q_rad):
    """Set the arm to a specific joint configuration (position, zero velocity)."""
    data.qpos[:7] = q_rad
    data.qvel[:7] = 0.0
    mujoco.mj_forward(model, data)


def simulate_transition(model, data, start_rad, target_rad,
                        max_steps=3000, tol=0.02, viewer=None):
    """Simulate start → target. Returns True if no collision and converged."""
    saved_mocap = data.mocap_pos.copy() if model.nmocap > 0 else None
    data.time = 0.0
    set_arm_state(model, data, start_rad)
    if saved_mocap is not None:
        data.mocap_pos[:] = saved_mocap
    data.ctrl[:7] = target_rad

    for _ in range(max_steps):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if data.ncon > 0:
            return False
        if (np.max(np.abs(data.qpos[:7] - target_rad)) < tol
                and np.max(np.abs(data.qvel[:7])) < 0.05):
            return True

    return np.max(np.abs(data.qpos[:7] - target_rad)) < tol


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
    """Greedy nearest-neighbor ordering. Returns index array."""
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

    candidates = latin_hypercube(args.n_candidates, rng)

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
    keep = [True] * len(valid)
    current_pos = np.zeros(7)
    prev_ee = get_ee_pos(model, data, current_pos) if viewer else None

    try:
        for i, target in enumerate(valid):
            if viewer is not None and not viewer.is_running():
                print("Viewer closed, stopping.")
                break

            if simulate_transition(model, data, current_pos, target,
                                   viewer=viewer):
                ee = data.xpos[ee_body_id].copy()
                d = joint_distance(current_pos, target)
                if viewer is not None:
                    add_line(viewer, prev_ee, ee, GREEN)
                    viewer.sync()
                    prev_ee = ee
                current_pos = target.copy()
                print(f"  [{i+1:3d}/{len(valid)}] OK   dist={d:.2f}")
            else:
                keep[i] = False
                print(f"  [{i+1:3d}/{len(valid)}] SKIP (collision in transit)")
    finally:
        if viewer is not None:
            viewer.close()

    final = valid[keep]
    n_dropped = len(valid) - len(final)

    # Prepend zero pose as the first pose
    q_rad = np.vstack([np.zeros(7), final])
    q_deg = np.rad2deg(q_rad)

    output = {
        "n_poses": int(len(q_deg)),
        "q_deg": q_deg.tolist(),
    }
    with open(args.output, "w") as f:
        yaml.dump(output, f, default_flow_style=None, sort_keys=False)

    print(f"\n{len(q_rad)} final poses (incl. zero) "
          f"({n_dropped} dropped in transit validation). "
          f"Saved to {args.output}")


if __name__ == "__main__":
    main()
