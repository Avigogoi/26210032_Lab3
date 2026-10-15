
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

MODEL = "scene.xml"

ARM_JOINTS = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6"
]

TCP_SITE = "right_center"


def object_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"Missing MuJoCo object: {name}")
    return idx


def get_arm_indices(model):
    joint_ids, qadr, dadr, actuator_ids = [], [], [], []

    for name in ARM_JOINTS:
        jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_ids.append(jid)
        qadr.append(model.jnt_qposadr[jid])
        dadr.append(model.jnt_dofadr[jid])

        aid = -1
        for a in range(model.nu):
            if (model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                    and model.actuator_trnid[a, 0] == jid):
                aid = a
                break

        if aid < 0:
            raise ValueError(f"No motor actuator for {name}")

        actuator_ids.append(aid)

    return (np.array(joint_ids), np.array(qadr),
            np.array(dadr), np.array(actuator_ids))


def rotation_error(R, R_target):
    # Rotation vector that takes the current orientation toward target.
    R_err = R_target @ R.T
    cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-8:
        return np.zeros(3)

    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ]) / (2.0 * np.sin(angle))

    return angle * axis


def solve_ik(model, data, site_id, target_pos,
             target_rot, qadr, dadr, max_steps=1000):

    q_initial = data.qpos[qadr].copy()
    damping = 0.03

    for step in range(max_steps):
        mujoco.mj_forward(model, data)

        pos = data.site_xpos[site_id].copy()
        error = np.asarray(target_pos, dtype=float) - pos
        err_norm = np.linalg.norm(error)

        if step % 100 == 0:
            print(
                f"IK {step}: error={err_norm:.4f} m, "
                f"TCP={np.round(pos, 4)}, "
                f"target={np.round(target_pos, 4)}"
            )

        if err_norm < 0.008:
            q_solution = data.qpos[qadr].copy()
            data.qpos[qadr] = q_initial
            mujoco.mj_forward(model, data)
            print(f"IK converged: error={err_norm:.4f} m")
            return q_solution, True

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(
            model, data, jacp, jacr, site_id
        )

        J = jacp[:, dadr]

        # Damped least-squares position update
        A = J @ J.T + damping**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(A, error)

        # Limit the update for stability
        dq_norm = np.linalg.norm(dq)
        if dq_norm > 0.04:
            dq *= 0.04 / dq_norm

        data.qpos[qadr] += dq

        # Respect the arm joint limits
        for k, name in enumerate(ARM_JOINTS):
            jid = object_id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            lo, hi = model.jnt_range[jid]
            data.qpos[qadr[k]] = np.clip(
                data.qpos[qadr[k]], lo + 1e-3, hi - 1e-3
            )

    data.qpos[qadr] = q_initial
    mujoco.mj_forward(model, data)

    print("IK failed to reach target.")
    return q_initial, False



def set_gripper(model, data, opening, steps, qadr, dadr,
                arm_actuators, viewer=None):

    # 0.025 m = fully open
    # About 0.002 m = closed enough to grip a 44 mm cube
    opening = float(np.clip(opening, 0.0, 0.025))

    left = object_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR,
        "left_finger_act"
    )
    right = object_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR,
        "right_finger_act"
    )

    data.ctrl[left] = opening
    data.ctrl[right] = opening

    q_hold = data.qpos[qadr].copy()

    for _ in range(steps):
        mujoco.mj_forward(model, data)

        for k, aid in enumerate(arm_actuators):
            torque = (
                data.qfrc_bias[dadr[k]]
                + 250.0 * (q_hold[k] - data.qpos[qadr[k]])
                - 35.0 * data.qvel[dadr[k]]
            )
            data.ctrl[aid] = torque

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()



def move_tcp(model, data, site_id, target, target_rot,
             qadr, dadr, arm_actuators, viewer=None):
    q_target, ok = solve_ik(
        model, data, site_id, np.asarray(target),
        target_rot, qadr, dadr
    )

    if not ok:
        print("Unreachable TCP target:", target)
        return False

    # Track the desired arm configuration using motor torques.
    start = data.time
    duration = 4.0

    while data.time - start < duration:
        mujoco.mj_forward(model, data)

        for k, aid in enumerate(arm_actuators):
            torque = (
                data.qfrc_bias[dadr[k]]
                + 250.0 * (q_target[k] - data.qpos[qadr[k]])
                - 35.0 * data.qvel[dadr[k]]
            )
            data.ctrl[aid] = torque

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

    mujoco.mj_forward(model, data)
    actual = data.site_xpos[site_id].copy()
    err = np.linalg.norm(actual - target)

    print("Target TCP:", np.round(target, 3))
    print("Actual TCP:", np.round(actual, 3))
    print(f"Tracking error: {err:.3f} m")

    return err < 0.02


def run(seed=0, show=True):
    rng = np.random.default_rng(seed)

    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    _, qadr, dadr, arm_actuators = get_arm_indices(model)
    site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    cube_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_qadr = model.jnt_qposadr[cube_joint]

    # Cube dimensions are 44 mm wide. Table top is at z = 0.20 m.
    x = float(rng.uniform(0.25, 0.39))
    y = float(rng.uniform(0.25, 0.39))
    yaw = float(rng.uniform(-np.pi, np.pi))

    data.qpos[cube_qadr:cube_qadr + 3] = [x, y, 0.222]
    data.qpos[cube_qadr + 3:cube_qadr + 7] = [
        np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)
    ]
    mujoco.mj_forward(model, data)

    viewer_context = (
        mujoco.viewer.launch_passive(model, data) if show else None
    )

    def execute(viewer):
        # Start with open fingers and hold the current arm pose.
        set_gripper(model, data, 0.002, 250,
                    qadr, dadr, arm_actuators, viewer)
                    
          

        # Use the current tool orientation for the initial diagnostic.
        

        mujoco.mj_forward(model, data)
        target_rot = data.site_xmat[site_id].reshape(3, 3).copy()
        mujoco.mj_forward(model, data)

	print("Initial TCP position:",
      		np.round(data.site_xpos[site_id], 4))

	print("Initial TCP rotation:")
	print(data.site_xmat[site_id].reshape(3, 3))

	print("Cube position:",
     		 np.round(data.xpos[cube_id], 4))
	
        # This sequence assumes a downward-facing gripper. If the
        # orientation is not suitable, calibrate target_rot first.
        targets = [
    	([x, y, 0.34], "pregrasp"),
    	([x, y, 0.265], "grasp"),
    	([x, y, 0.36], "lift"),
    	([0.54, 0.40, 0.36], "above_tray"),
    	([0.54, 0.40, 0.29], "place")
          ]


        for pos, label in targets:
            print("\nStage:", label)

            if not move_tcp(
                model, data, site_id, pos, target_rot,
                qadr, dadr, arm_actuators, viewer
            ):
                return False

            if label == "grasp":
    set_gripper(
        model, data, 0.002, 250,
        qadr, dadr, arm_actuators, viewer
    )

    mujoco.mj_forward(model, data)

    print(
        "Cube position after closing:",
        np.round(data.xpos[cube_id], 4)
    )

elif label == "place":
    set_gripper(
        model, data, 0.025, 250,
        qadr, dadr, arm_actuators, viewer
    )

        mujoco.mj_forward(model, data)
        cube_pos = data.xpos[cube_id].copy()

        # The cube should finish inside the tray's approximate bounds.
        placed = (
            0.465 < cube_pos[0] < 0.615
            and 0.33 < cube_pos[1] < 0.47
            and 0.21 < cube_pos[2] < 0.32
        )

        print("\nCube final position:", np.round(cube_pos, 4))
        print("Placement success:", placed)
        return placed

    if viewer_context is None:
        success = execute(None)
    else:
        with viewer_context as viewer:
            success = execute(viewer)

    print({
        "seed": seed,
        "success": bool(success),
        "cube_initial": [x, y, yaw]
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    run(args.seed, show=not args.no_viewer)

