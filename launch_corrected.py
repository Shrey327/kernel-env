"""Fresh100-dollar experiment: source-bound gates and bounded owned-app cleanup."""
import argparse,json,subprocess,sys,time,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'train'))
from corrected_contract import RUN,digest,source_files,require_gate
from spend_guard import SpendGuard


def stopped(app_id):
    p=subprocess.run(['modal','app','list','--json'],capture_output=True,text=True,timeout=10,check=True)
    row=next((r for r in json.loads(p.stdout) if r.get('App ID')==app_id),None)
    return bool(row and row.get('State')=='stopped' and str(row.get('Tasks'))=='0')


def cleanup(app_id,call,success,stop_app=None,is_stopped=stopped,cancel_timeout=5):
    errors=[]
    if call is not None and not success:
        def cancel():
            try:call.cancel(terminate_containers=True)
            except Exception as e:errors.append(type(e).__name__)
        thread=threading.Thread(target=cancel,daemon=True);thread.start();thread.join(cancel_timeout)
        if thread.is_alive():errors.append('cancel_timeout; independent app stop follows')
    if stop_app is None:
        stop_app=lambda:subprocess.run(['modal','app','stop',app_id],capture_output=True,text=True,timeout=15,check=True)
    stop_app()
    until=time.monotonic()+20
    while not is_stopped(app_id):
        if time.monotonic()>=until:raise RuntimeError('owned app stop/zero workers unconfirmed')
        time.sleep(2)
    return {'app_id':app_id,'zero_workers_verified':True,'cancel_errors':errors}


def check_launch(phase,contract,local,gates):
    if contract.get('run_id')!=RUN:raise ValueError('wrong corrected identity')
    require_gate(local,contract,'local')
    if phase!='validate':
        for stage in ('gpu-validation','reward-calibration','sft-verification'):
            require_gate(gates.get(stage,{}),contract,stage)
        if phase=='train':require_gate(gates.get('pilot',{}),contract,'pilot')
        if phase=='train' or phase.startswith('eval-'):
            require_gate(gates.get('independent-review',{}),contract,'independent-review')
        if phase=='eval-rl':require_gate(gates.get('train',{}),contract,'train')


def no_progress_seconds(phase):
    # One 600s original-shape eval (or validation case) can legitimately run without a progress update.
    return 660 if phase=='validate' or phase.startswith('eval-') else 300


def progress(reservation):
    path=RUN+'/progress/'+reservation+'.json'
    code='import modal,sys;sys.stdout.buffer.write(b"".join(modal.Volume.from_name("kernel-env-runs").read_file(sys.argv[1])))'
    p=subprocess.run([sys.executable,'-c',code,path],capture_output=True,timeout=10)
    return json.loads(p.stdout) if p.returncode==0 else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('validate','pilot','train','eval-base','eval-sft','eval-rl'),required=True)
    parser.add_argument('--ledger',required=True);parser.add_argument('--contract',required=True);parser.add_argument('--local-gate',required=True)
    parser.add_argument('--gates')
    args=parser.parse_args();root=Path(__file__).parent
    contract=json.loads(Path(args.contract).read_text());local=json.loads(Path(args.local_gate).read_text())
    if source_files(root)!=contract['source_files']:raise ValueError('local source changed after preflight')
    gates=json.loads(Path(args.gates).read_text()) if args.gates else {}
    check_launch(args.phase,contract,local,gates)
    ledger=json.loads(Path(args.ledger).read_text())
    if ledger.get('new_run_id')!=RUN or ledger.get('authorization_baseline_utc')!='2026-10-19T16:01:11Z':raise ValueError('wrong fresh budget ledger')
    envelope,seconds={'validate':('validation',1800),'pilot':('pilot',1800),'train':('training',9000),
        'eval-base':('evaluation',3600),'eval-sft':('evaluation',3600),'eval-rl':('evaluation',3600)}[args.phase]
    guard=SpendGuard(args.ledger);reservation=guard.reserve('corrected-'+args.phase,envelope,seconds)
    guard.record(reservation['reservation_id'],source_digest=contract['source_digest'],protocol_digest=contract['protocol_digest'],run_id=RUN)
    print(json.dumps({'reserved':reservation}),flush=True)
    from modal_corrected import app,validate,stage
    call=None;app_id=None;success=False;result=None;error=None
    try:
        with app.run(detach=True):
            app_id=app.app_id
            guard.record(reservation['reservation_id'],state='uncertain',app_id=app_id,reason='spawn in progress; reconcile before retry')
            call=(validate.spawn(reservation,contract,json.loads(Path(args.ledger).read_text())) if args.phase=='validate' else
                  stage.spawn(args.phase,reservation,contract,json.loads(Path(args.ledger).read_text()),gates))
            guard.record(reservation['reservation_id'],state='launched',call_id=call.object_id)
            print(json.dumps({'app_id':app_id,'call_id':call.object_id,'reservation_id':reservation['reservation_id']}),flush=True)
            startup=time.time()+90;last_progress=time.time();last_signature=None;body=False
            while True:
                if time.time()>=reservation['deadline_unix']-20:raise TimeoutError('absolute validation safety cutoff')
                info=progress(reservation['reservation_id'])
                if info:
                    body=True;signature=info.get('semantic_progress')
                    if signature!=last_signature:last_progress=time.time();last_signature=signature
                if not body and time.time()>=startup:raise TimeoutError('90-second startup bound')
                if time.time()-last_progress>no_progress_seconds(args.phase):raise TimeoutError('stage no-progress bound')
                try:result=call.get(timeout=0)
                except TimeoutError:time.sleep(5);continue
                success=bool(result.get('passed'));break
    except BaseException as e:error=e
    finally:
        if app_id:
            try:proof=cleanup(app_id,call,success)
            except BaseException as e:
                guard.record(reservation['reservation_id'],state='uncertain',cleanup_error=str(e)[:500]);raise
        else:proof={'no_app_created':True}
        guard.record(reservation['reservation_id'],state='complete' if success else 'failed',result=result,
                     error=str(error)[:500] if error else None,finished_unix=time.time(),cleanup=proof)
    if error:raise error
    if not success:raise RuntimeError('corrected GPU preflight failed; preserve result, no retry')
    print(json.dumps({'completed':result}),flush=True)


if __name__=='__main__':main()
