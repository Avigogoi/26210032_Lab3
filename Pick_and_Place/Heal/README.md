# HEAL Pick-and-Place in MuJoCo (starter project)

This project adapts the uploaded HEAL arm description and replaces missing, machine-specific Robotiq mesh paths with a simple primitive parallel-jaw gripper. It includes a scene with table, cube, tray, DLS controller, workspace sampling, batch CSV logging, and a QP velocity-step helper.

## Dependencies

Python 3.10+ recommended. In a virtual environment:

```bash
python -m pip install -r requirements.txt
```

## Files

- `scene.xml`: combined HEAL model + primitive gripper + table/cube/tray.
- `scripts/inspect_model.py`: list joint/site/actuator names.
- `scripts/pick_place.py`: initial DLS pick-and-place controller.
- `scripts/episode_runner.py`: batch runs and CSV logging.
- `scripts/workspace_analysis.py`: sampled workspace plot.
- `scripts/dls_ik.py`, `scripts/qp_ik.py`: reusable IK steps.
- `scripts/mink_baseline.py`: Mink installation/model smoke test; task setup must be completed against the installed Mink API version.

## Run

From this folder:

```bash
python scripts/inspect_model.py
python scripts/pick_place.py --seed 0
python scripts/workspace_analysis.py
python scripts/episode_runner.py --episodes 25 --no-viewer
```

Run batch mode without a viewer for overnight experiments. Results are saved in `results/episode_logs.csv` and `results/workspace.png`.

## Important limitations / required validation

1. The uploaded ZIP does not contain the referenced Robotiq `assets/*.stl` files, and `arm_gripper.xml` has absolute paths tied to another computer. The project therefore uses a primitive gripper rather than the original Robotiq 2F-85 geometry.
2. The HEAL TCP orientation, table coordinates, and pick/place heights are initial guesses. First run `inspect_model.py`, then use the viewer to verify the robot's reachable area and tune targets.
3. The included `pick_place.py` is a starting controller, not a validated success-guaranteed manipulation pipeline. A physical grasp depends on finger orientation, collision geometry, friction, control gains, and the robot's workspace. Do not claim a successful grasp until the cube is visibly lifted and placed.
4. Workspace orientation filtering is approximate. A proper dexterous workspace should run IK at each sampled position for all required yaw angles and include collision checks.
5. The CSV schema includes placeholders for clearance and IK iterations; instrument these before reporting them as measured metrics.
6. The Mink file is a smoke test, not yet a complete Mink pick/place implementation. The DLS baseline is included to establish a custom IK path first; finish and validate Mink task setup before comparing all three methods.
7. The QP helper needs a compatible solver backend; joint position, velocity, and collision constraints must be explicitly built into the full controller.

## Report outline

1. Problem statement and robot model.
2. Scene geometry, gripper design, and table dimensions.
3. FK, Jacobian, and IK methods (Mink, DLS, QP).
4. Task and dexterous workspace sampling methodology.
5. Pick/place sequence and episode randomization.
6. Metrics and reproducible seed protocol.
7. Results: success rate, planning time, clearance, iterations, limit violations.
8. Failure screenshots and categorized causes.
9. Comparison, limitations, and improvements.
