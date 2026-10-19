"""Explicit shared reservation, one paid stage, and a foreground absolute-deadline guard."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'train'))
from spend_guard import SpendGuard

PHASES={'smoke':('validation_warmup_pilot',120),'validate':('validation_warmup_pilot',1200),'validate-h100':('validation_warmup_pilot',1200),
 'warmup':('validation_warmup_pilot',1200),'pilot':('validation_warmup_pilot',1500),
 'train':('training',5400),'eval-base':('evaluation',2400),'eval-sft':('evaluation',2400),'eval-rl':('evaluation',2400)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=list(PHASES),required=True)
    parser.add_argument('--ledger',required=True);args=parser.parse_args()
    hold=Path(args.ledger).with_name('evaluation-hold.json')
    if args.phase.startswith('eval-') and hold.exists() and json.loads(hold.read_text()).get('active'):
        raise RuntimeError('Evaluation held before reservation: '+json.loads(hold.read_text()).get('reason','timing audit'))
    guard=SpendGuard(args.ledger);envelope,seconds=PHASES[args.phase]
    reservation=guard.reserve(args.phase,envelope,seconds)
    root=Path(__file__).parent
    paths=[*root.joinpath('kernel_env').rglob('*.py'),*root.joinpath('train').rglob('*.py'),root/'modal_better.py',root/'modal_app.py',root/'launch_better.py']
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if '.venv' not in p.parts and '__pycache__' not in p.parts}
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    reservation=guard.record(reservation['reservation_id'],source_digest=digest,source_files=hashes)
    print(json.dumps({'reserved':{k:v for k,v in reservation.items() if k!='source_files'}}),flush=True)
    from modal_better import app,validate,validate_h100,warmup,experiment_stage,import_smoke
    call=None
    try:
        with app.run(detach=True):
            if time.time()>reservation['deadline_unix']-60:raise TimeoutError('launch preparation used stage time')
            budget=json.loads(Path(args.ledger).read_text())
            if args.phase=='smoke':call=import_smoke.spawn(reservation['deadline_unix'],reservation)
            elif args.phase in ('validate','validate-h100'):
                function=validate if args.phase=='validate' else validate_h100
                call=function.spawn(reservation['deadline_unix'],reservation)
            elif args.phase=='warmup':call=warmup.spawn(reservation['deadline_unix'],reservation,budget)
            else:call=experiment_stage.spawn(args.phase,reservation['deadline_unix'],reservation,budget)
            row=guard.record(reservation['reservation_id'],state='launched',app_id=app.app_id,call_id=call.object_id)
            print(json.dumps({'launched':{k:v for k,v in row.items() if k!='source_files'}}),flush=True)
            startup_deadline=min(time.time()+90,reservation['deadline_unix']-30)
            body_started=False
            while True:
                if not body_started:
                    try:
                        marker=b''.join(__import__('modal').Volume.from_name('kernel-env-runs').read_file('better-20261019/started/'+reservation['reservation_id']+'.json'))
                        body_started=bool(json.loads(marker).get('body_started_unix'))
                    except FileNotFoundError:pass
                if time.time()>=reservation['deadline_unix']-30 or (not body_started and time.time()>=startup_deadline):
                    call.cancel(terminate_containers=True)
                    guard.record(reservation['reservation_id'],state='cancel_requested',reason='stage safety cutoff or 90-second startup bound')
                    # The app belongs only to this single reserved stage. Stop it too: queued inputs may remain listed after cancel.
                    result=subprocess.run(['modal','app','stop',app.app_id],capture_output=True,text=True,timeout=20)
                    if result.returncode:raise RuntimeError('call cancelled but app stop unconfirmed')
                    guard.record(reservation['reservation_id'],state='cancelled',finished_unix=time.time())
                    raise TimeoutError('absolute stage deadline')
                try:result=call.get(timeout=0)
                except TimeoutError:time.sleep(min(5,max(0,reservation['deadline_unix']-30-time.time())));continue
                guard.record(reservation['reservation_id'],state='complete',result=result,finished_unix=time.time())
                print(json.dumps({'completed':result}),flush=True);break
    except BaseException as error:
        current=next(r for r in json.loads(Path(args.ledger).read_text())['new_jobs'] if r['reservation_id']==reservation['reservation_id'])
        if current['state'] not in ('cancelled','complete'):
            guard.record(reservation['reservation_id'],state='uncertain',
                         launch_error=type(error).__name__,error=str(error)[:1500],finished_unix=time.time())
        raise


if __name__=='__main__':main()
