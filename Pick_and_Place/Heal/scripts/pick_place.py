
#!/usr/bin/env python3

import argparse
import os
import sys
import time

import mujoco
import numpy as np

try:
    import mujoco.viewer
except ImportError:
    mujoco_viewer_available = False
else:
    mujoco_viewer_available = True


# ============================================================
# CONFIGURATION: MATCHES YOUR scene.xml
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_FILE = os.path.join(ROOT, "scene.xml")

ARM_JOINTS = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]

ARM_ACTUATORS = [
    "turret", "shoulder", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]

GRIPPER_ACTUATORS = [
    "left_finger_act",
    "right_finger_act",
]

TCP_SITE = "right_center"
CUBE_BODY = "cube"
CUBE_JOINT = "cube_free"

TABLE_TOP = 0.200
CUBE_HALF = 0.022

TRAY_X = 0.540
TRAY_Y = 0.400

# Torque PD controller. Tune if the arm oscillates or moves slowly.
KP = 100.0
KD = 75.0

# IK settings
DAMPING = 0.05
IK_ITERATIONS = 400
IK_TOLERANCE = 0.008
MAX_JOINT_STEP = 0.08


# ============================================================
# NAME LOOKUP
# ============================================================

def get_id(model, object_type, name):
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise ValueError(
            f"Missing MuJoCo object '{name}'. "
            "Check the name in scene.xml."
        )
    return result


def get_joint_info(model, names):
    joint_ids = []
    qpos_addresses = []
    dof_addresses = []

    for name in names:
        jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_ids.append(jid)
        qpos_addresses.append(int(model.jnt_qposadr[jid]))
        dof_addresses.append(int(model.jnt_dofadr[jid]))

    return (
        np.array(joint_ids, dtype=int),
        np.array(qpos_addresses, dtype=int),
        np.array(dof_addresses, dtype=int),
    )


def get_joint_limits(model, joint_ids):
    lower = []
    upper = []

    for jid in joint_ids:
        if model.jnt_limited[jid]:
            lower.append(model.jnt_range[jid, 0] + 1e-4)
            upper.append(model.jnt_range[jid, 1] - 1e-4)
        else:
            lower.append(-2.0 * np.pi)
            upper.append(2.0 * np.pi)

    return np.array(lower), np.array(upper)


# ============================================================
# DAMPED LEAST-SQUARES INVERSE KINEMATICS
# ============================================================

def solve_ik(
    model, data, tcp_id,
    qpos_addresses, dof_addresses,
    q_lower, q_upper, target
):
    """
    Position-only damped least-squares IK.

    Returns:
        best joint configuration,
        final position error in metres,
        iteration count.
    """
    target = np.asarray(target, dtype=float)
    saved_qpos = data.qpos.copy()

    q = data.qpos[qpos_addresses].copy()
    best_q = q.copy()
    best_error = float("inf")
    iterations_used = 0

    try:
        for iteration in range(IK_ITERATIONS):
            iterations_used = iteration + 1

            data.qpos[qpos_addresses] = q
            mujoco.mj_forward(model, data)

            current = data.site_xpos[tcp_id].copy()
            error = target - current
            error_norm = float(np.linalg.norm(error))

            if error_norm < best_error:
                best_error = error_norm
                best_q = q.copy()

            if error_norm < IK_TOLERANCE:
                break

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))

            mujoco.mj_jacSite(
                model, data, jacp, jacr, tcp_id
            )

            J = jacp[:, dof_addresses]

            # DLS update: dq = J.T (J J.T + lambda^2 I)^-1 e
            matrix = (
                J @ J.T
                + DAMPING**2 * np.eye(3)
            )

            dq = J.T @ np.linalg.solve(matrix, error)

            max_step = np.max(np.abs(dq))
            if max_step > MAX_JOINT_STEP:
                dq *= MAX_JOINT_STEP / max_step

            q = np.clip(q + dq, q_lower, q_upper)

        data.qpos[qpos_addresses] = best_q
        mujoco.mj_forward(model, data)

        final_position = data.site_xpos[tcp_id].copy()
        final_error = float(np.linalg.norm(target - final_position))

        return best_q.copy(), final_error, iterations_used

    finally:
        # Restore the simulation state used before the IK calculation.
        data.qpos[:] = saved_qpos
        mujoco.mj_forward(model, data)


# ============================================================
# ARM AND GRIPPER CONTROL
# ============================================================

def command_arm(
    model, data, actuator_ids,
    qpos_addresses, dof_addresses, q_target
):
    """
    Torque PD control with gravity/Coriolis bias compensation.
    The six arm actuators in your XML are <motor> actuators.
    """
    q = data.qpos[qpos_addresses]
    dq = data.qvel[dof_addresses]

    error = q_target - q
    bias = data.qfrc_bias[dof_addresses]

    torque = KP * error - KD * dq + bias

    for i, aid in enumerate(actuator_ids):
        if model.actuator_ctrllimited[aid]:
            lo, hi = model.actuator_ctrlrange[aid]
            torque[i] = np.clip(torque[i], lo, hi)

        data.ctrl[aid] = torque[i]


def set_gripper(model, data, gripper_ids, opening):
    """
    The XML position actuators accept values from 0 to 0.025 m.
    0.025 = open; 0.000 = closed.
    """
    opening = float(np.clip(opening, 0.0, 0.025))

    for aid in gripper_ids:
        if model.actuator_ctrllimited[aid]:
            lo, hi = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = np.clip(opening, lo, hi)
        else:
            data.ctrl[aid] = opening


def step_for(
    model, data, arm_ids,
    qpos_addresses, dof_addresses,
    q_target, duration, viewer=None,
    gripper_ids=None, gripper_opening=None
):
    steps = max(1, int(duration / model.opt.timestep))

    for _ in range(steps):
        command_arm(
            model, data, arm_ids,
            qpos_addresses, dof_addresses, q_target
        )

        if gripper_ids is not None and gripper_opening is not None:
            set_gripper(
                model, data, gripper_ids, gripper_opening
            )

        mujoco.mj_step(model, data)

        if viewer is not None:
            if not viewer.is_running():
                return False
            viewer.sync()

    return True


# ============================================================
# MOVE TCP TO A POSITION
# ============================================================

def move_tcp(
    model, data, tcp_id,
    arm_ids, qpos_addresses, dof_addresses,
    q_lower, q_upper, target, duration, viewer=None
):
    target = np.asarray(target, dtype=float)

    q_solution, ik_error, iterations = solve_ik(
        model, data, tcp_id,
        qpos_addresses, dof_addresses,
        q_lower, q_upper, target
    )

    print(
        f"Target {target.round(3)} | "
        f"IK error {ik_error:.4f} m | "
        f"{iterations} iterations"
    )

    if ik_error > 0.05:
        print(
            "WARNING: target may be unreachable. "
            "Check the workspace and TCP site."
        )

    ok = step_for(
        model, data, arm_ids,
        qpos_addresses, dof_addresses,
        q_solution, duration, viewer
    )

    actual = data.site_xpos[tcp_id].copy()
    tracking_error = float(np.linalg.norm(target - actual))

    print(
        f"  TCP actual={actual.round(3)}; "
        f"tracking error={tracking_error:.4f} m"
    )

    return ok, tracking_error


# ============================================================
# CUBE RESET AND PICK-AND-PLACE
# ============================================================

def place_cube(model, data, rng, cube_qpos, cube_dof):
    x = float(rng.uniform(0.28, 0.35))
    y = float(rng.uniform(0.28, 0.36))
    yaw = float(rng.uniform(-np.pi, np.pi))

    # Free joint qpos: x, y, z, qw, qx, qy, qz
    data.qpos[cube_qpos:cube_qpos + 7] = [
        x,
        y,
        TABLE_TOP + CUBE_HALF,
        np.cos(yaw / 2.0),
        0.0,
        0.0,
        np.sin(yaw / 2.0),
    ]

    data.qvel[cube_dof:cube_dof + 6] = 0.0
    mujoco.mj_forward(model, data)

    return np.array([x, y, TABLE_TOP + CUBE_HALF])


def run_episode(seed=0, show=True):
    # mujoco is imported at module scope, avoiding UnboundLocalError.
    if not os.path.isfile(XML_FILE):
        raise FileNotFoundError(f"XML file not found: {XML_FILE}")

    rng = np.random.default_rng(seed)

    model = mujoco.MjModel.from_xml_path(XML_FILE)
    data = mujoco.MjData(model)

    # Resolve all named objects.
    tcp_id = get_id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    cube_body_id = get_id(
        model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY
    )
    cube_joint_id = get_id(
        model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT
    )

    cube_qpos = int(model.jnt_qposadr[cube_joint_id])
    cube_dof = int(model.jnt_dofadr[cube_joint_id])

    (
        arm_joint_ids,
        qpos_addresses,
        dof_addresses,
    ) = get_joint_info(model, ARM_JOINTS)

    q_lower, q_upper = get_joint_limits(model, arm_joint_ids)

    arm_ids = np.array([
        get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in ARM_ACTUATORS
    ], dtype=int)

    gripper_ids = np.array([
        get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in GRIPPER_ACTUATORS
    ], dtype=int)

    mujoco.mj_resetData(model, data)

    cube_initial = place_cube(
        model, data, rng, cube_qpos, cube_dof
    )

    # Keep the robot at the joint configuration defined in the XML.
    mujoco.mj_forward(model, data)

    viewer = None

    print("\n========================================")
    print("HEAL PICK-AND-PLACE")
    print("========================================")
    print(f"XML: {XML_FILE}")
    print(f"Seed: {seed}")
    print(f"DOFs: {model.nv}")
    print(f"Timestep: {model.opt.timestep:.4f} s")
    print(f"Initial cube position: {cube_initial.round(3)}")

    try:
        if show:
            if not mujoco_viewer_available:
                raise RuntimeError(
                    "MuJoCo viewer unavailable. "
                    "Run with --no-viewer or install a supported "
                    "MuJoCo package."
                )
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.sync()

        # Let the cube settle.
        for _ in range(int(0.4 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            if viewer is not None:
                if not viewer.is_running():
                    return False
                viewer.sync()

        cube_position = data.xpos[cube_body_id].copy()
        print(f"Settled cube position: {cube_position.round(3)}")

        # Initial gripper opening.
        set_gripper(model, data, gripper_ids, 0.025)

        # TCP target positions. These are starting values, not a
        # guarantee that the gripper fingers align with the cube.
        pregrasp = np.array([
            cube_position[0], cube_position[1], 0.36
        ])
        grasp = np.array([
            cube_position[0], cube_position[1], 0.25
        ])
        lift = np.array([
            cube_position[0], cube_position[1], 0.38
        ])
        above_tray = np.array([
            TRAY_X, TRAY_Y, 0.38
        ])
        place = np.array([
            TRAY_X, TRAY_Y, 0.29
        ])

        print("\n[1/6] Approach cube")
        ok, _ = move_tcp(
            model, data, tcp_id, arm_ids,
            qpos_addresses, dof_addresses,
            q_lower, q_upper, pregrasp, 2.0, viewer
        )
        if not ok:
            return False

        print("\n[2/6] Descend")
        ok, _ = move_tcp(
            model, data, tcp_id, arm_ids,
            qpos_addresses, dof_addresses,
            q_lower, q_upper, grasp, 1.5, viewer
        )
        if not ok:
            return False

        print("\n[3/6] Close gripper")
        q_hold = data.qpos[qpos_addresses].copy()
        ok = step_for(
            model, data, arm_ids,
            qpos_addresses, dof_addresses,
            q_hold, 0.8, viewer,
            gripper_ids, 0.0
        )
        if not ok:
            return False

        print("\n[4/6] Lift cube")
        ok, _ = move_tcp(
            model, data, tcp_id, arm_ids,
            qpos_addresses, dof_addresses,
            q_lower, q_upper, lift, 1.8, viewer
        )
        if not ok:
            return False

        print("\n[5/6] Move above tray")
        ok, _ = move_tcp(
            model, data, tcp_id, arm_ids,
            qpos_addresses, dof_addresses,
            q_lower, q_upper, above_tray, 2.0, viewer
        )
        if not ok:
            return False

        print("\n[6/6] Lower into tray")
        ok, _ = move_tcp(
            model, data, tcp_id, arm_ids,
            qpos_addresses, dof_addresses,
            q_lower, q_upper, place, 1.5, viewer
        )
        if not ok:
            return False

        print("\nOpen gripper and release cube")
        q_hold = data.qpos[qpos_addresses].copy()
        ok = step_for(
            model, data, arm_ids,
            qpos_addresses, dof_addresses,
            q_hold, 0.8, viewer,
            gripper_ids, 0.025
        )
        if not ok:
            return False

        # Let the released cube settle.
        for _ in range(int(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()

        final_cube = data.xpos[cube_body_id].copy()

        inside_tray = (
            TRAY_X - 0.060 <= final_cube[0] <= TRAY_X + 0.060
            and TRAY_Y - 0.055 <= final_cube[1] <= TRAY_Y + 0.055
            and final_cube[2] >= 0.20
        )

        print("\n========================================")
        print(f"Final cube position: {final_cube.round(4)}")
        print(f"Cube within tray bounds: {inside_tray}")
        print(f"RESULT: {'SUCCESS' if inside_tray else 'FAILURE'}")
        print("========================================")

        # Keep the graphical viewer open until manually closed.
        if viewer is not None:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)

        return bool(inside_tray)

    finally:
        if viewer is not None:
            viewer.close()


def main():
    parser = argparse.ArgumentParser(
        description="HEAL arm MuJoCo pick-and-place"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run without the graphical viewer",
    )
    args = parser.parse_args()

    success = run_episode(
        seed=args.seed,
        show=not args.no_viewer,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

