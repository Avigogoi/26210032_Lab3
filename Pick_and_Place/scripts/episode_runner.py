"""Run N deterministic episodes and save per-episode CSV logs."""
import os, csv, argparse, time
from pick_place import run_episode

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--episodes',type=int,default=25); ap.add_argument('--start-seed',type=int,default=0); ap.add_argument('--no-viewer',action='store_true'); args=ap.parse_args()
    out=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','results')); os.makedirs(out,exist_ok=True)
    path=os.path.join(out,'episode_logs.csv'); rows=[]
    for i in range(args.episodes):
        seed=args.start_seed+i; t0=time.perf_counter()
        try: success,reason=run_episode(seed,not args.no_viewer)
        except Exception as e: success,reason=False,'exception:'+repr(e)
        rows.append({'episode_id':i,'seed':seed,'ik_method':'DLS-baseline','success':int(success),'failure_reason':reason,'planning_time_s':time.perf_counter()-t0,'ik_iterations':'not instrumented','min_clearance_m':'not instrumented','joint_limit_violations':'not instrumented','velocity_limit_violations':'not instrumented'})
        with open(path,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        print(f'Episode {i+1}/{args.episodes}: success={success}, reason={reason}')
    rate=sum(r['success'] for r in rows)/len(rows) if rows else 0
    print(f'Success rate: {rate:.1%}; logs: {path}')
if __name__=='__main__': main()
