"""Joint-space random sampling for an approximate HEAL TCP workspace plot."""
import os
import numpy as np
import matplotlib.pyplot as plt
import mujoco
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); model=mujoco.MjModel.from_xml_path(os.path.join(ROOT,'scene.xml')); data=mujoco.MjData(model)
jnames=[f'joint_{i}' for i in range(1,7)]; jids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in jnames]; qadr=[model.jnt_qposadr[j] for j in jids]; site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'right_center')
rng=np.random.default_rng(7); points=[]; dex=[]
for _ in range(12000):
    for j,qa in zip(jids,qadr): data.qpos[qa]=rng.uniform(*model.jnt_range[j])
    mujoco.mj_forward(model,data); p=data.site_xpos[site].copy(); points.append(p)
    # Approximate dexterous subset: required nominal downward TCP orientation near identity; tighten by task checks in full report.
    R=data.site_xmat[site].reshape(3,3); score=np.linalg.norm(R-np.eye(3))
    if score<0.7: dex.append(p)
points=np.asarray(points); dex=np.asarray(dex)
fig=plt.figure(); ax=fig.add_subplot(111,projection='3d'); ax.scatter(points[:,0],points[:,1],points[:,2],s=1,alpha=.12,label='Task candidates')
if len(dex): ax.scatter(dex[:,0],dex[:,1],dex[:,2],s=2,alpha=.5,label='Orientation-filtered candidates')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)'); ax.legend(); ax.set_title('HEAL sampled TCP workspace');
out=os.path.join(ROOT,'results'); os.makedirs(out,exist_ok=True); fig.savefig(os.path.join(out,'workspace.png'),dpi=180); plt.show()
