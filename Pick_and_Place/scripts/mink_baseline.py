"""Mink API smoke test. Complete FrameTask setup after verifying installed Mink version."""
import mujoco, mink, os
MODEL=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','scene.xml'))
model=mujoco.MjModel.from_xml_path(MODEL)
configuration=mink.Configuration(model)
print('Mink loaded with model. q size:', configuration.q.size)
print('Next: create a FrameTask for site right_center and solve each target with a supported QP solver.')
