
import argparse
import csv
import time
import numpy as np
import mujoco
import mujoco.viewer

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTS = [f"act_joint{i}" for i in range(1, 8)]
TABLE_Z = 0.72
CUBE_HALF = 0.025
TRAY_XY = np.array([0.72, 0.28])


def get_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"Model is missing: {name}")
    return idx


def solve_dls(model, data, site_id, target,
              qadr, dadr, lower, upper,
              iterations=250, damping=0.08):
    """Position-only damped least-squares IK."""
    original = data.qpos[qadr].copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    best_q = original.copy()
    best_error = float("inf")

    for it in range(iterations):
        mujoco.mj_forward(model, data)
        error = target - data.site_xpos[site_id]
        err_norm = np.linalg.norm(error)

        if err_norm < best_error:
            best_error = err_norm
            best_q = data.qpos[qadr].copy()

        if err_norm < 0.012:
            data.qpos[qadr] = original
            mujoco.mj_forward(model, data)
            return best_q, True, it + 1

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = jacp[:, dadr]

        # dq = J.T @ inv(J @ J.T + lambda^2 I) @ error
        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3), error
        )
        dq = np.clip(dq, -0.06, 0.06)

        data.qpos[qadr] = np.clip(
            data.qpos[qadr] + dq, lower + 0.005, upper - 0.005
        )

    data.qpos[qadr] = original
    mujoco.mj_forward(model, data)
    return best_q, best_error < 0.012, iterations


def move_to(model, data, viewer, site_id, target,
            qadr, dadr, lower, upper, act_ids,
            duration=1.2):
    q_target, ok, iters = solve_dls(
        model, data, site_id, np.asarray(target),
        qadr, dadr, lower, upper
    )

    if not ok:
        return False, iters, float("inf")

    data.ctrl[act_ids] = q_target
    steps = int(duration / model.opt.timestep)

    for _ in range(steps):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(model.opt.timestep)

    mujoco.mj_forward(model, data)
    error = np.linalg.norm(target - data.site_xpos[site_id])
    return error < 0.045, iters, float(error)


def move_gripper(model, data, viewer, left_id, right_id,
                 command, duration=0.4):
    data.ctrl[left_id] = command
    data.ctrl[right_id] = command

    for _ in range(int(duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(model.opt.timestep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--log", default="episode_log.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    qadr, dadr, lower, upper, arm_act_ids = [], [], [], [], []

    for jname, aname in zip(ARM_JOINTS, ARM_ACTS):
        jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        aid = get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        qadr.append(model.jnt_qposadr[jid])
        dadr.append(model.jnt_dofadr[jid])
        lower.append(model.jnt_range[jid, 0])
        upper.append(model.jnt_range[jid, 1])
        arm_act_ids.append(aid)

    qadr, dadr = np.array(qadr), np.array(dadr)
    lower, upper = np.array(lower), np.array(upper)
    arm_act_ids = np.array(arm_act_ids)

    tcp_id = get_id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    cube_jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_adr = model.jnt_qposadr[cube_jid]
    left_id = get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_left")
    right_id = get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_right")

    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [0.45, 0.05, 0.85]

    rows = []

    try:
        for ep in range(args.episodes):
            mujoco.mj_resetData(model, data)

            # Random cube pose on tabletop.
            x = rng.uniform(0.30, 0.48)
            y = rng.uniform(-0.30, -0.08)
            yaw = rng.uniform(-np.pi, np.pi)

            data.qpos[cube_adr:cube_adr + 3] = [
                x, y, TABLE_Z + CUBE_HALF + 0.002
            ]
            data.qpos[cube_adr + 3:cube_adr + 7] = [
                np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)
            ]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

            # Fingers start open; command maximum slide to close.
            move_gripper(model, data, viewer, left_id, right_id,
                         0.0, 0.2)

            cube_top = TABLE_Z + 2 * CUBE_HALF
            total_iters = 0
            reason = ""
            success = False
            start = time.perf_counter()

            # Positions are TCP targets in world coordinates.
            phases = [
                ("pregrasp", [x, y, cube_top + 0.14]),
                ("grasp",    [x, y, cube_top + 0.025]),
                ("lift",     [x, y, cube_top + 0.17]),
                ("transit",  [TRAY_XY[0], TRAY_XY[1],
                              cube_top + 0.22]),
                ("place",    [TRAY_XY[0], TRAY_XY[1], 0.84]),
            ]

            # Reach the grasp pose first.
            for phase, target in phases[:2]:
                ok, iters, err = move_to(
                    model, data, viewer, tcp_id, np.array(target),
                    qadr, dadr, lower, upper, arm_act_ids
                )
                total_iters += iters
                if not ok:
                    reason = f"IK/position tracking failed at {phase}; error={err:.3f} m"
                    break

            if not reason:
                # Close gripper. Real grasp depends on contact geometry.
                move_gripper(model, data, viewer, left_id, right_id,
                             0.025, 0.5)

                for phase, target in phases[2:]:
                    ok, iters, err = move_to(
                        model, data, viewer, tcp_id, np.array(target),
                        qadr, dadr, lower, upper, arm_act_ids
                    )
                    total_iters += iters
                    if not ok:
                        reason = f"IK/position tracking failed at {phase}; error={err:.3f} m"
                        break

            if not reason:
                move_gripper(model, data, viewer, left_id, right_id,
                             0.0, 0.5)

                for _ in range(250):
                    mujoco.mj_step(model, data)
                    if viewer is not None:
                        viewer.sync()
                        time.sleep(model.opt.timestep)

                final_cube = data.qpos[cube_adr:cube_adr + 3].copy()
                success = (
                    0.56 < final_cube[0] < 0.88
                    and 0.14 < final_cube[1] < 0.42
                    and final_cube[2] > TABLE_Z
                )
                if not success:
                    reason = (
                        "Cube not detected inside tray; final position="
                        + str(np.round(final_cube, 3).tolist())
                    )
            else:
                final_cube = data.qpos[cube_adr:cube_adr + 3].copy()

            elapsed = time.perf_counter() - start
            print(
                f"Episode {ep + 1}: {'SUCCESS' if success else 'FAIL'} | "
                f"cube=({x:.3f},{y:.3f}), yaw={yaw:.2f} | "
                f"IK iterations={total_iters} | {elapsed:.2f}s | "
                f"{reason or 'placed'}"
            )

            rows.append({
                "episode": ep + 1,
                "seed": args.seed,
                "cube_x": x,
                "cube_y": y,
                "cube_yaw": yaw,
                "success": int(success),
                "failure_reason": reason,
                "time_s": elapsed,
                "ik_iterations": total_iters,
            })

    finally:
        if viewer is not None:
            viewer.close()

    with open(args.log, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    rate = np.mean([r["success"] for r in rows]) if rows else 0
    print(f"\nSuccess rate: {100 * rate:.1f}%")
    print(f"CSV log: {args.log}")


if __name__ == "__main__":
    main()

