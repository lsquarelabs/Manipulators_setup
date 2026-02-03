#!/usr/bin/env python3
"""Generate collision-free joint configurations for gravity sysid.

Loads the Kinova Gen3 MuJoCo model with a table, generates candidate
configurations via Latin Hypercube Sampling, then builds a sequential
chain using nearest-neighbor ordering: from the current pose, pick the
closest untested candidate, simulate the direct transition, and accept
if collision-free. The output sequence can be executed on the real robot
without returning to home between poses.

No ROS required.

Usage:
    python generate_poses.py --n-candidates 500 --n-poses 80 --output poses.npz
"""

import os
import argparse
import numpy as np
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


def joint_distance(a, b):
    """Max absolute joint difference (Chebyshev distance)."""
    return np.max(np.abs(a - b))


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
    p.add_argument("--n-poses", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="poses.npz")
    p.add_argument("--visualize", action="store_true",
                   help="Open MuJoCo viewer to watch validation")
    p.add_argument("--light", action="store_true",
                   help="Add lights and floor to the scene")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.xml} ...")
    n_green = args.n_poses if args.visualize else 0
    n_red = args.n_candidates if args.visualize else 0
    model = load_scene(args.xml, n_green=n_green, n_red=n_red, light=args.light)
    data = mujoco.MjData(model)

    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    "end_effector_link")

    candidates = latin_hypercube(args.n_candidates, rng)
    available = np.ones(args.n_candidates, dtype=bool)  # mask of unused candidates

    viewer = None
    if args.visualize:
        viewer = mujoco.viewer.launch_passive(model, data)

    valid = []
    via_home = []  # True if this pose required a home detour
    n_fail = 0
    checked = 0
    zero_pos = np.zeros(7)
    current_pos = zero_pos.copy()
    # EE positions for line drawing
    prev_ee = get_ee_pos(model, data, current_pos) if viewer else None
    zero_ee = prev_ee  # cache zero-pose EE position

    GREEN = np.array([0, 255, 0, 180], dtype=np.uint8)
    YELLOW = np.array([255, 255, 0, 180], dtype=np.uint8)

    print(f"Building chain from {args.n_candidates} candidates for {args.n_poses} valid ...")

    try:
        while len(valid) < args.n_poses and np.any(available):
            if viewer is not None and not viewer.is_running():
                print("Viewer closed, stopping.")
                break

            # Find nearest available candidate to current position
            dists = np.full(args.n_candidates, np.inf)
            avail_idx = np.where(available)[0]
            for idx in avail_idx:
                dists[idx] = joint_distance(current_pos, candidates[idx])
            order = np.argsort(dists)

            advanced = False
            for idx in order:
                if not available[idx]:
                    continue
                target = candidates[idx]
                available[idx] = False
                checked += 1

                # Try direct: current → target
                if simulate_transition(model, data, current_pos, target,
                                       viewer=viewer):
                    ee = data.xpos[ee_body_id].copy()
                    if viewer is not None:
                        data.mocap_pos[len(valid)] = ee
                        add_line(viewer, prev_ee, ee, GREEN)
                        viewer.sync()
                    d = joint_distance(current_pos, target)
                    valid.append(target)
                    via_home.append(False)
                    current_pos = target.copy()
                    prev_ee = ee
                    print(f"  [{len(valid):3d}/{args.n_poses}] candidate {idx+1:4d} OK  "
                          f"dist={d:.2f}  "
                          f"[{', '.join(f'{a:+.2f}' for a in target)}]")
                    advanced = True
                    break

                # Try via home: zero → target
                if simulate_transition(model, data, zero_pos, target,
                                       viewer=viewer):
                    ee = data.xpos[ee_body_id].copy()
                    if viewer is not None:
                        data.mocap_pos[len(valid)] = ee
                        add_line(viewer, prev_ee, zero_ee, YELLOW)
                        add_line(viewer, zero_ee, ee, YELLOW)
                        viewer.sync()
                    d = joint_distance(zero_pos, target)
                    valid.append(target)
                    via_home.append(True)
                    current_pos = target.copy()
                    prev_ee = ee
                    print(f"  [{len(valid):3d}/{args.n_poses}] candidate {idx+1:4d} OK  "
                          f"(via home) dist={d:.2f}  "
                          f"[{', '.join(f'{a:+.2f}' for a in target)}]")
                    advanced = True
                    break

                # Both failed — skip, stay at current_pos
                if viewer is not None and n_fail < n_red:
                    data.mocap_pos[n_green + n_fail] = data.xpos[ee_body_id].copy()
                    viewer.sync()
                n_fail += 1
    finally:
        if viewer is not None:
            viewer.close()

    if len(valid) < args.n_poses:
        print(f"\nWARNING: only {len(valid)}/{args.n_poses} valid from "
              f"{checked} candidates. Increase --n-candidates.")
    n_detours = sum(via_home)
    if n_detours:
        print(f"  {n_detours} poses required a home detour.")

    q_rad = np.array(valid)
    q_deg_kinova = np.rad2deg(q_rad)

    np.savez(args.output, q_rad=q_rad, q_deg_kinova=q_deg_kinova,
             via_home=np.array(via_home))

    rate = len(valid) / checked * 100 if checked else 0
    print(f"\n{len(valid)} valid poses ({rate:.0f}% acceptance). Saved to {args.output}")


if __name__ == "__main__":
    main()
