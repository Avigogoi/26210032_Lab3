
#!/usr/bin/env python3

from pathlib import Path
import argparse

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "scene.xml"

ARM_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]

ARM_ACTUATOR_NAMES = [
    "turret", "shoulder", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]

TCP_SITE_NAME = "right_center"
CUBE_BODY_NAME = "cube"

LEFT_GRIPPER_ACTUATOR = "left_finger_act"
RIGHT_GRIPPER_ACTUATOR = "right_finger_act"

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.025

# Initial PD gains. Diagnostics below report the actual tracking.
KP = 180.0
KD = 35.0

# Conservative per-joint torque cap, in N*m.
# The XML motor control range is separately respected.
MAX_TORQUE = 250.0

IK_MAX_STEPS = 1500
IK_DAMPING = 0.08
IK_MAX_DQ = 0.06
IK_POSITION_TOL = 0.008
IK_ORIENTATION_TOL = 0.15

MOVE_DURATION = 3.0
SETTLE_TIME = 1.0

GRASP_TCP_OFFSET = 0.050
PREGRASP_HEIGHT = 0.12
LIFT_HEIGHT = 0.15

TRAY_X = 0.54
TRAY_Y = 0.40
PLACE_TCP_Z = 0.295

MIN_LIFT_HEIGHT = 0.025


# ============================================================
# MODEL LOOKUP
# ============================================================

def get_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)

    if object_id < 0:
        raise ValueError(
            f"Cannot find '{name}' in scene.xml. "
            "Check the exact name in your XML."
        )

    return object_id


def get_arm_indices(model):
    joint_ids = []
    qpos_addresses = []
    dof_addresses = []

    for name in ARM_JOINT_NAMES:
        jid = get_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )

        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"{name} is not a hinge joint.")

        joint_ids.append(jid)
        qpos_addresses.append(int(model.jnt_qposadr[jid]))
        dof_addresses.append(int(model.jnt_dofadr[jid]))

    return (
        np.asarray(joint_ids, dtype=int),
        np.asarray(qpos_addresses, dtype=int),
        np.asarray(dof_addresses, dtype=int),
    )


def get_actuator_ids(model):
    arm_ids = [
        get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in ARM_ACTUATOR_NAMES
    ]

    left_id = get_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR,
        LEFT_GRIPPER_ACTUATOR,
    )

    right_id = get_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR,
        RIGHT_GRIPPER_ACTUATOR,
    )

    return np.asarray(arm_ids, dtype=int), left_id, right_id


def check_actuator_mapping(model, actuator_ids, joint_ids):
    print("\nARM ACTUATOR MAPPING")

    for i, aid in enumerate(actuator_ids):
        actual_joint = int(model.actuator_trnid[aid, 0])
        expected_joint = int(joint_ids[i])

        print(
            f"  {ARM_ACTUATOR_NAMES[i]:10s} -> "
            f"{ARM_JOINT_NAMES[i]:10s}, "
            f"gear={model.actuator_gear[aid, 0]:.3f}"
        )

        if actual_joint != expected_joint:
            raise ValueError(
                f"Actuator '{ARM_ACTUATOR_NAMES[i]}' is mapped "
                f"to the wrong joint."
            )

    print("Actuator mapping verified.")


# ============================================================
# TCP AND CUBE STATE
# ============================================================

def tcp_position(data, site_id):
    return data.site_xpos[site_id].copy()


def tcp_rotation(data, site_id):
    return data.site_xmat[site_id].reshape(3, 3).copy()


def cube_position(data, body_id):
    return data.xpos[body_id].copy()


# ============================================================
# ARM CONTROLLER
# ============================================================

def apply_arm_controller(
    model,
    data,
    actuator_ids,
    qpos_addresses,
    dof_addresses,
    q_reference,
):
    q = data.qpos[qpos_addresses]
    qd = data.qvel[dof_addresses]

    # Joint PD torque plus MuJoCo bias-force compensation.
    desired_torque = (
        data.qfrc_bias[dof_addresses]
        + KP * (q_reference - q)
        - KD * qd
    )

    desired_torque = np.clip(
        desired_torque, -MAX_TORQUE, MAX_TORQUE
    )

    for i, aid in enumerate(actuator_ids):
        gear = float(model.actuator_gear[aid, 0])

        if abs(gear) < 1e-12:
            data.ctrl[aid] = 0.0
            continue

        # For a simple motor actuator, joint torque is
        # approximately gear * actuator control.
        control = desired_torque[i] / gear

        if model.actuator_ctrllimited[aid]:
            low, high = model.actuator_ctrlrange[aid]
            control = np.clip(control, low, high)

        data.ctrl[aid] = control


def step_simulation(model, data, viewer=None):
    mujoco.mj_step(model, data)

    if viewer is not None:
        viewer.sync()


def simulate_hold(
    model,
    data,
    actuator_ids,
    qpos_addresses,
    dof_addresses,
    duration,
    viewer=None,
    q_reference=None,
):
    if q_reference is None:
        q_reference = data.qpos[qpos_addresses].copy()

    steps = max(1, int(duration / model.opt.timestep))

    for _ in range(steps):
        apply_arm_controller(
            model, data, actuator_ids,
            qpos_addresses, dof_addresses,
            q_reference,
        )
        step_simulation(model, data, viewer)


def track_joint_target(
    model,
    data,
    actuator_ids,
    qpos_addresses,
    dof_addresses,
    q_target,
    viewer=None,
    duration=MOVE_DURATION,
):
    q_start = data.qpos[qpos_addresses].copy()
    steps = max(1, int(duration / model.opt.timestep))

    # Smooth cubic interpolation.
    for step in range(steps):
        s = (step + 1) / steps
        blend = 3.0 * s**2 - 2.0 * s**3
        q_reference = q_start + blend * (q_target - q_start)

        apply_arm_controller(
            model, data, actuator_ids,
            qpos_addresses, dof_addresses,
            q_reference,
        )
        step_simulation(model, data, viewer)

    simulate_hold(
        model, data, actuator_ids,
        qpos_addresses, dof_addresses,
        0.25, viewer, q_target,
    )

    # --------------------------------------------------------
    # DIAGNOSTICS: actual versus desired joint configuration
    # --------------------------------------------------------
    q_actual = data.qpos[qpos_addresses].copy()
    q_velocity = data.qvel[dof_addresses].copy()
    joint_error = q_target - q_actual
    total_error = np.linalg.norm(joint_error)

    print("\nJOINT TRACKING DIAGNOSTICS")
    print(
        f"{'Joint':<12}"
        f"{'Target(rad)':>14}"
        f"{'Actual(rad)':>14}"
        f"{'Error(rad)':>14}"
        f"{'Velocity':>12}"
    )

    for i, name in enumerate(ARM_JOINT_NAMES):
        print(
            f"{name:<12}"
            f"{q_target[i]:>14.4f}"
            f"{q_actual[i]:>14.4f}"
            f"{joint_error[i]:>14.4f}"
            f"{q_velocity[i]:>12.4f}"
        )

    print("\nACTUATOR CONTROL VALUES")

    for i, aid in enumerate(actuator_ids):
        if model.actuator_ctrllimited[aid]:
            control_range = model.actuator_ctrlrange[aid]
        else:
            control_range = np.array([-np.inf, np.inf])

        print(
            f"{ARM_ACTUATOR_NAMES[i]:<12}"
            f"ctrl={data.ctrl[aid]:>10.3f} "
            f"range=[{control_range[0]:.1f}, "
            f"{control_range[1]:.1f}]"
        )

    print(f"\nTotal joint tracking error: {total_error:.4f} rad")

    return total_error < 0.40


# ============================================================
# INVERSE KINEMATICS
# ============================================================

def orientation_error(R, R_desired):
    return 0.5 * (
        np.cross(R[:, 0], R_desired[:, 0])
        + np.cross(R[:, 1], R_desired[:, 1])
        + np.cross(R[:, 2], R_desired[:, 2])
    )


def solve_ik(
    model,
    data,
    site_id,
    target_position,
    target_rotation,
    joint_ids,
    qpos_addresses,
    dof_addresses,
):
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()

    q = data.qpos[qpos_addresses].copy()
    best_q = q.copy()
    best_score = float("inf")
    best_pe = float("inf")
    best_re = float("inf")

    target_position = np.asarray(target_position, dtype=float)
    target_rotation = np.asarray(target_rotation).reshape(3, 3)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    try:
        for iteration in range(IK_MAX_STEPS):
            data.qpos[qpos_addresses] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            position = tcp_position(data, site_id)
            rotation = tcp_rotation(data, site_id)

            ep = target_position - position
            er = orientation_error(rotation, target_rotation)

            pe = np.linalg.norm(ep)
            re = np.linalg.norm(er)
            score = pe + 0.10 * re

            if score < best_score:
                best_score = score
                best_q = q.copy()
                best_pe = pe
                best_re = re

            if pe < IK_POSITION_TOL and re < IK_ORIENTATION_TOL:
                print(
                    f"IK converged at iteration {iteration + 1}: "
                    f"position error={pe:.4f} m, "
                    f"orientation error={re:.4f} rad"
                )
                return best_q

            mujoco.mj_jacSite(
                model, data, jacp, jacr, site_id
            )

            Jp = jacp[:, dof_addresses]
            Jr = jacr[:, dof_addresses]

            weight = 0.25
            J = np.vstack((Jp, weight * Jr))
            error = np.concatenate((ep, weight * er))

            lhs = J @ J.T + IK_DAMPING**2 * np.eye(6)

            try:
                dq = J.T @ np.linalg.solve(lhs, error)
            except np.linalg.LinAlgError:
                dq = J.T @ np.linalg.pinv(lhs) @ error

            dq = np.clip(dq, -IK_MAX_DQ, IK_MAX_DQ)
            q += dq

            for i, jid in enumerate(joint_ids):
                if model.jnt_limited[jid]:
                    low, high = model.jnt_range[jid]
                    q[i] = np.clip(q[i], low + 1e-4, high - 1e-4)

        print(
            "IK iteration limit reached. "
            f"Best position error={best_pe:.4f} m, "
            f"orientation error={best_re:.4f} rad"
        )

        return best_q if best_pe <= 0.035 else None

    finally:
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(model, data)


def move_tcp(
    model,
    data,
    site_id,
    target_position,
    target_rotation,
    joint_ids,
    qpos_addresses,
    dof_addresses,
    actuator_ids,
    label,
    viewer=None,
):
    print(f"\nMOVE: {label}")
    print("Target TCP:", np.round(target_position, 4))

    q_target = solve_ik(
        model, data, site_id,
        target_position, target_rotation,
        joint_ids, qpos_addresses, dof_addresses,
    )

    if q_target is None:
        print(f"IK failed for '{label}'.")
        return False

    tracking_ok = track_joint_target(
        model, data, actuator_ids,
        qpos_addresses, dof_addresses,
        q_target, viewer,
    )

    actual = tcp_position(data, site_id)
    position_error = np.linalg.norm(target_position - actual)

    print("Actual TCP:", np.round(actual, 4))
    print(f"TCP position error: {position_error:.4f} m")

    return tracking_ok and position_error < 0.035


# ============================================================
# GRIPPER
# ============================================================

def set_gripper(
    model,
    data,
    arm_actuator_ids,
    left_id,
    right_id,
    qpos_addresses,
    dof_addresses,
    target,
    viewer=None,
    duration=0.5,
):
    target = float(np.clip(target, GRIPPER_OPEN, GRIPPER_CLOSED))
    arm_reference = data.qpos[qpos_addresses].copy()

    steps = max(1, int(duration / model.opt.timestep))

    for _ in range(steps):
        apply_arm_controller(
            model, data, arm_actuator_ids,
            qpos_addresses, dof_addresses,
            arm_reference,
        )

        data.ctrl[left_id] = target
        data.ctrl[right_id] = target

        step_simulation(model, data, viewer)

    print(f"Gripper command={target:.3f}")


# ============================================================
# PICK AND PLACE
# ============================================================

def run_episode(model, data, viewer=None, seed=0):
    np.random.seed(seed)

    joint_ids, qpos_addresses, dof_addresses = get_arm_indices(model)
    actuator_ids, left_id, right_id = get_actuator_ids(model)

    site_id = get_id(
        model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE_NAME
    )
    cube_id = get_id(
        model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_NAME
    )

    check_actuator_mapping(model, actuator_ids, joint_ids)

    mujoco.mj_forward(model, data)

    q_home = data.qpos[qpos_addresses].copy()

    data.ctrl[left_id] = GRIPPER_OPEN
    data.ctrl[right_id] = GRIPPER_OPEN

    print("Initial TCP:", np.round(tcp_position(data, site_id), 4))
    print("Settling cube...")

    simulate_hold(
        model, data, actuator_ids,
        qpos_addresses, dof_addresses,
        SETTLE_TIME, viewer, q_home,
    )

    cube_start = cube_position(data, cube_id).copy()
    desired_rotation = tcp_rotation(data, site_id)

    print("Cube position:", np.round(cube_start, 4))

    # 1. Open gripper.
    set_gripper(
        model, data, actuator_ids,
        left_id, right_id, qpos_addresses, dof_addresses,
        GRIPPER_OPEN, viewer,
    )

    # 2. Move above the cube.
    pregrasp = np.array([
        cube_start[0],
        cube_start[1],
        cube_start[2] + GRASP_TCP_OFFSET + PREGRASP_HEIGHT,
    ])

    if not move_tcp(
        model, data, site_id, pregrasp, desired_rotation,
        joint_ids, qpos_addresses, dof_addresses,
        actuator_ids, "above cube", viewer,
    ):
        print("Stopping because the above-cube target was not reached.")
        return False

    # 3. Descend to the grasp position.
    grasp = np.array([
        cube_start[0],
        cube_start[1],
        cube_start[2] + GRASP_TCP_OFFSET,
    ])

    if not move_tcp(
        model, data, site_id, grasp, desired_rotation,
        joint_ids, qpos_addresses, dof_addresses,
        actuator_ids, "grasp cube", viewer,
    ):
        print("Stopping because the grasp target was not reached.")
        return False

    # 4. Close fingers.
    set_gripper(
        model, data, actuator_ids,
        left_id, right_id, qpos_addresses, dof_addresses,
        GRIPPER_CLOSED, viewer, duration=0.8,
    )

    cube_before_lift = cube_position(data, cube_id).copy()

    # 5. Lift.
    lift = grasp.copy()
    lift[2] += LIFT_HEIGHT

    if not move_tcp(
        model, data, site_id, lift, desired_rotation,
        joint_ids, qpos_addresses, dof_addresses,
        actuator_ids, "lift cube", viewer,
    ):
        print("The lift target was not reached.")
        return False

    simulate_hold(
        model, data, actuator_ids,
        qpos_addresses, dof_addresses,
        0.4, viewer,
    )

    cube_after_lift = cube_position(data, cube_id).copy()
    lift_amount = cube_after_lift[2] - cube_before_lift[2]

    print(f"Cube vertical movement: {lift_amount:.4f} m")

    if lift_amount < MIN_LIFT_HEIGHT:
        print(
            "PICK FAILED: cube did not rise. Check TCP alignment, "
            "finger motion, contacts, and friction."
        )
        return False

    print("PICK SUCCESS")

    # 6. Move above tray.
    above_tray = np.array([
        TRAY_X, TRAY_Y, PLACE_TCP_Z + 0.10
    ])

    if not move_tcp(
        model, data, site_id, above_tray, desired_rotation,
        joint_ids, qpos_addresses, dof_addresses,
        actuator_ids, "above tray", viewer,
    ):
        print("Cube was picked, but the above-tray target failed.")
        return False

    # 7. Lower into tray.
    place = np.array([TRAY_X, TRAY_Y, PLACE_TCP_Z])

    if not move_tcp(
        model, data, site_id, place, desired_rotation,
        joint_ids, qpos_addresses, dof_addresses,
        actuator_ids, "place in tray", viewer,
    ):
        print("Cube was picked, but the placement target failed.")
        return False

    # 8. Release cube.
    set_gripper(
        model, data, actuator_ids,
        left_id, right_id, qpos_addresses, dof_addresses,
        GRIPPER_OPEN, viewer, duration=0.7,
    )

    simulate_hold(
        model, data, actuator_ids,
        qpos_addresses, dof_addresses,
        1.0, viewer,
    )

    # 9. Verify placement.
    final_cube = cube_position(data, cube_id)

    print("\nFINAL CUBE POSITION:", np.round(final_cube, 4))

    inside_x = abs(final_cube[0] - TRAY_X) < 0.037
    inside_y = abs(final_cube[1] - TRAY_Y) < 0.032
    acceptable_z = 0.215 < final_cube[2] < 0.29

    success = inside_x and inside_y and acceptable_z

    print(
        "PICK-AND-PLACE SUCCESS"
        if success else
        "Cube released, but its final position is outside "
        "the approximate tray region."
    )

    return success


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="HEAL manipulator pick-and-place"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run without the graphical viewer",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Create the model and data only once.
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    if args.no_viewer:
        success = run_episode(model, data, None, args.seed)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.sync()
            success = run_episode(
                model, data, viewer, args.seed
            )
            viewer.sync()

            # Keep the final simulation state visible.
            while viewer.is_running():
                viewer.sync()

    print("\nEpisode success:", success)


if __name__ == "__main__":
    main()

