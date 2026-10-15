"""QP IK velocity step; requires qpsolvers plus a backend such as clarabel/osqp."""
import numpy as np
from qpsolvers import solve_qp

def qp_ik_step(J, twist, qdot_min, qdot_max, damping=1e-3, solver='clarabel'):
    J=np.asarray(J,float); twist=np.asarray(twist,float)
    # min 0.5*qdot.T P qdot + q.T qdot, subject to bounds
    P=J.T@J + damping*np.eye(J.shape[1]); q=-(J.T@twist)
    sol=solve_qp(P,q,lb=np.asarray(qdot_min),ub=np.asarray(qdot_max),solver=solver)
    if sol is None: raise RuntimeError('QP IK infeasible or solver unavailable')
    return sol
