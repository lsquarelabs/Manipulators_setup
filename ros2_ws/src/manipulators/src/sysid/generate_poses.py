#!/usr/bin/env python3
"""Generate collision-free joint configurations for gravity sysid.

Loads the Kinova Gen3 MuJoCo model with a table, generates candidate
configurations, simulates the full motion path from home to each target,
and keeps only poses reachable without any collision (table or self).

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
    <global offwidth="1920" offheight="1080"/>
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


def is_pose_valid(model, data, target_rad, home_key_id,
                  max_steps=3000, tol=0.02, viewer=None):
    """Simulate home → target. Returns True if no collision and converged.
    Exits early once converged (no need to keep simulating)."""
    saved_mocap = data.mocap_pos.copy() if model.nmocap > 0 else None
    mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    if saved_mocap is not None:
        data.mocap_pos[:] = saved_mocap
    data.ctrl[:7] = target_rad

    for _ in range(max_steps):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if data.ncon > 0:
            return False
        # Early exit once converged and velocity is low
        if (np.max(np.abs(data.qpos[:7] - target_rad)) < tol
                and np.max(np.abs(data.qvel[:7])) < 0.05):
            return True

    # Timed out — check if it at least converged
    return np.max(np.abs(data.qpos[:7] - target_rad)) < tol


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

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise RuntimeError("'home' keyframe not found")

    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    "end_effector_link")

    candidates = latin_hypercube(args.n_candidates, rng)

    viewer = None
    if args.visualize:
        viewer = mujoco.viewer.launch_passive(model, data)

    valid = []
    n_fail = 0
    checked = 0
    print(f"Checking {args.n_candidates} candidates for {args.n_poses} valid ...")

    try:
        for i, target in enumerate(candidates):
            if len(valid) >= args.n_poses:
                break
            if viewer is not None and not viewer.is_running():
                print("Viewer closed, stopping.")
                break
            checked += 1
            if is_pose_valid(model, data, target, home_id, viewer=viewer):
                if viewer is not None:
                    data.mocap_pos[len(valid)] = data.xpos[ee_body_id].copy()
                    viewer.sync()
                valid.append(target)
                print(f"  [{len(valid):3d}/{args.n_poses}] candidate {i+1:4d} OK  "
                      f"[{', '.join(f'{a:+.2f}' for a in target)}]")
            else:
                if viewer is not None:
                    data.mocap_pos[n_green + n_fail] = data.xpos[ee_body_id].copy()
                    viewer.sync()
                    n_fail += 1
    finally:
        if viewer is not None:
            viewer.close()

    if len(valid) < args.n_poses:
        print(f"\nWARNING: only {len(valid)}/{args.n_poses} valid from "
              f"{checked} candidates. Increase --n-candidates.")

    q_rad = np.array(valid)
    q_deg_kinova = np.rad2deg(q_rad)

    np.savez(args.output, q_rad=q_rad, q_deg_kinova=q_deg_kinova)

    rate = len(valid) / checked * 100 if checked else 0
    print(f"\n{len(valid)} valid poses ({rate:.0f}% acceptance). Saved to {args.output}")


if __name__ == "__main__":
    main()
