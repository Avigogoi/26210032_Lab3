import os, mujoco
MODEL=os.path.join(os.path.dirname(__file__), '..', 'scene.xml')
model=mujoco.MjModel.from_xml_path(MODEL)
data=mujoco.MjData(model)
print('Loaded model:', model.nq, 'qpos,', model.nv, 'dofs,', model.nu, 'actuators')
for i in range(model.njnt):
    print('JOINT', i, mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i), 'range=',model.jnt_range[i])
for i in range(model.nu):
    print('ACTUATOR',i,mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i))
for name in ['right_center','cube_geom']:
    typ=mujoco.mjtObj.mjOBJ_SITE if name=='right_center' else mujoco.mjtObj.mjOBJ_GEOM
    print(name, mujoco.mj_name2id(model,typ,name))
