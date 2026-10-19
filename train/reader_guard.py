"""Backup for the existing private reader. Run only after its guard handoff is coordinated."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

APP_ID='ap-sXAmPnyHHyFO65EBoXbkIm'
SERVICE='kernel-env-better-dashboard'
EXPIRES_UNIX=1792442951.508414
CUTOFF_UNIX=EXPIRES_UNIX-30
MARKERS=tuple(run+'/'+name for run in ('better-20261019','corrected-20261019')
              for name in ('comparison.json','TERMINAL.json'))


def validate_policy(policy):
    if (policy.get('app_id')!=APP_ID or policy.get('service_name')!=SERVICE or
        policy.get('expires_unix')!=EXPIRES_UNIX or policy.get('created_unix')!=EXPIRES_UNIX-21600 or
        policy.get('max_cost_usd')!=0.5):
        raise ValueError('existing reader identity or original lease changed')


def stop_reason(now,terminal=False):
    return 'experiment terminal' if terminal else 'original lease cutoff' if now>=CUTOFF_UNIX else None


def terminal_marker():
    # A subprocess bounds SDK/network waits, so a stalled volume read cannot bypass expiry.
    code='''import modal,sys
v=modal.Volume.from_name('kernel-env-runs')
for path in sys.argv[1:]:
    try:
        next(iter(v.read_file(path)),b'')
        print(path)
        break
    except FileNotFoundError:pass
'''
    result=subprocess.run([sys.executable,'-c',code,*MARKERS],capture_output=True,text=True,
                          timeout=min(8,max(.1,CUTOFF_UNIX-time.time())),check=True)
    return result.stdout.strip() in MARKERS


def stopped(rows):
    row=next((r for r in rows if r.get('App ID')==APP_ID),None)
    return bool(row and row.get('State')=='stopped' and str(row.get('Tasks'))=='0')


def stop_and_verify(run=subprocess.run,monotonic=time.monotonic,sleep=time.sleep):
    deadline=monotonic()+28
    stop_error=None
    try:run(['modal','app','stop',APP_ID],capture_output=True,text=True,timeout=12,check=True)
    except (subprocess.SubprocessError,OSError) as error:stop_error=type(error).__name__
    while monotonic()<deadline:
        try:
            result=run(['modal','app','list','--json'],capture_output=True,text=True,
                       timeout=min(5,deadline-monotonic()),check=True)
            if stopped(json.loads(result.stdout)):
                return {'app_id':APP_ID,'zero_workers_verified':True,'stop_error':stop_error}
        except (subprocess.SubprocessError,OSError,ValueError):pass
        sleep(min(2,max(0,deadline-monotonic())))
    return {'app_id':APP_ID,'zero_workers_verified':False,'stop_error':stop_error,
            'outcome':'uncertain: app stopped with zero workers was not verified'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True);args=parser.parse_args()
    validate_policy(json.loads(args.policy.read_text()))
    while True:
        reason=stop_reason(time.time())
        if reason is None:
            try:reason=stop_reason(time.time(),terminal_marker())
            except (subprocess.SubprocessError,OSError):reason=stop_reason(time.time())
        if reason:
            result={**stop_and_verify(),'reason':reason,'at_unix':time.time(),'expires_unix':EXPIRES_UNIX}
            args.result.write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True)
            if not result['zero_workers_verified']:raise RuntimeError(result['outcome'])
            return
        time.sleep(min(20,max(0,CUTOFF_UNIX-time.time())))


if __name__=='__main__':main()
