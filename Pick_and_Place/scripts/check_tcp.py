import os
import mujoco
import numpy as np

root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
model = mujoco.MjModel.from_xml_path(
    os.path.join(root, "scene.xml")
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

site_name = "right_center"
sid = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, site_name
)

if sid < 0:
    raise ValueError(f"Missing TCP site: {site_name}")

print("Initial TCP position:", data.site_xpos[sid])
print(
    "Initial TCP rotation:\n",
    data.site_xmat[sid].reshape(3, 3)
)
print("Cube position:", data.xpos[
    mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "cube"
    )
])
