"""Reusable damped least-squares differential IK step."""
import numpy as np

def dls_step(J, error, damping=0.08, max_norm=0.08):
    J=np.asarray(J,float); e=np.asarray(error,float)
    dq=J.T @ np.linalg.solve(J@J.T+damping**2*np.eye(J.shape[0]),e)
    n=np.linalg.norm(dq)
    if n>max_norm: dq*=max_norm/n
    return dq
