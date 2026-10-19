from fastapi.testclient import TestClient
import better_dashboard as module


def test_existing_authentication_is_reused_and_missing_cost_stays_unknown(tmp_path,monkeypatch):
    import json,time
    now=time.time()
    (tmp_path/'reader-policy.json').write_text(json.dumps({'service_name':module.SERVICE_NAME,
        'created_unix':now,'expires_unix':now+3600,'max_cost_usd':0.5}))
    requests=[]
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Opener:
        def open(self,req,timeout):
            requests.append(req)
            assert req.get_header('X-kernel-dashboard-token')=='existing-token'
            return Response()
    monkeypatch.setattr(module.urllib.request,'build_opener',lambda *args:Opener())
    monkeypatch.setattr(module,'state_from',lambda *args:{'phase':'Preparing','activity':'waiting'})
    client=TestClient(module.make_app(tmp_path,lambda:None))
    assert client.get('/state').status_code==403
    result=client.get('/state',headers={'X-Kernel-Dashboard-Token':'existing-token'})
    assert result.status_code==200
    assert result.json()['cost'] is None
    assert result.json()['budget_cap_usd']==100
    assert len(requests)==1 and requests[0].full_url.startswith(module.EXISTING_READER)
    assert 'existing-token' not in result.text
    assert client.get('/metrics.json',headers={'X-Kernel-Dashboard-Token':'existing-token'}).status_code==200
    assert len(requests)==1


def test_reader_lease_is_bounded_and_terminal_data_stops_it():
    policy={'service_name':module.SERVICE_NAME,'created_unix':100,'expires_unix':100+21600,'max_cost_usd':0.5}
    assert not module.lease_expired(policy,101)
    assert module.lease_expired(policy,101,terminal=True)
    assert module.lease_expired(policy,21700)
    assert module.lease_expired({**policy,'expires_unix':21701},101)
    assert module.lease_expired({},101)


def test_recorded_submissions_and_missing_latency_are_truthful(tmp_path):
    import json
    path=tmp_path/'experiment/seed42/adapter';path.mkdir(parents=True)
    ep={'task_id':'relu','turns':[{'correct':True},{'correct':False}]}
    (path/'rollouts-rank0.jsonl').write_text(json.dumps(ep)+'\n')
    state=module.recorded_activity(tmp_path,{'phase':'Training'})
    assert state['kernel_evals']==2
    assert state['recent_evaluation']=={'calls':2,'queue_wait_s':None,'eval_s':None}
    assert '1 training episodes' in state['activity']


def test_sft_is_counted_and_groups_use_frozen_corrected_contract(tmp_path):
    import json
    (tmp_path/'evaluations').mkdir()
    ep={'task_id':'relu','turns':[{'stage':'timed','correct':True,'speedup':1.0}]}
    (tmp_path/'evaluations/sft.jsonl').write_text(json.dumps(ep)+'\n')
    (tmp_path/'contract.json').write_text(json.dumps({'protocol':{'evaluation_groups':{'heldout':['relu']}}}))
    state=module.recorded_activity(tmp_path,{'phase':'Evaluating'})
    assert state['kernel_evals']==1
    assert state['sft_groups']['heldout']['episodes']==1


def test_validation_history_preserves_failures_and_incomplete_attempts(tmp_path):
    import json
    failed=tmp_path/'validation/attempt-1';failed.mkdir(parents=True)
    (failed/'cases.jsonl').write_text(json.dumps({'case':'sensitivity','passed':False})+'\n')
    pending=tmp_path/'validation/attempt-2';pending.mkdir()
    (pending/'cases.jsonl').write_text(json.dumps({'case':'identity','passed':True})+'\n')
    states={r['reservation_id']:r for r in module.validation_attempts(tmp_path)}
    assert states['attempt-1']['state']=='failed'
    assert states['attempt-1']['failed_cases']==['sensitivity']
    assert states['attempt-2']['state']=='incomplete'
    assert not states['attempt-2']['complete']
    (pending/'cases.jsonl').write_text(json.dumps({'case':'old 2MiB diagnostic','passed':False,'required':False})+'\n')
    state=next(r for r in module.validation_attempts(tmp_path) if r['reservation_id']=='attempt-2')
    assert state['state']=='incomplete' and state['failed_cases']==['old 2MiB diagnostic']


def test_corrected_endpoint_exposes_failed_validation_without_claiming_training(tmp_path,monkeypatch):
    import json,time
    older=tmp_path/'better-20261019';older.mkdir()
    (older/'reader-policy.json').write_text(json.dumps({'service_name':module.SERVICE_NAME,
        'created_unix':time.time(),'expires_unix':time.time()+3600,'max_cost_usd':.5}))
    validation=tmp_path/'corrected-20261019/validation/attempt-2';validation.mkdir(parents=True)
    (validation/'cases.jsonl').write_text(json.dumps({'case':'reward calibration','passed':False})+'\n')
    (validation/'contract.json').write_text(json.dumps({'protocol_digest':'frozen-protocol'}))
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Opener:
        def open(self,*args,**kwargs):return Response()
    monkeypatch.setattr(module.urllib.request,'build_opener',lambda *args:Opener())
    monkeypatch.setattr(module,'state_from',lambda *args:{'phase':'Preparing','activity':'waiting'})
    result=TestClient(module.make_app(older,lambda:None)).get('/state?run=corrected',
        headers={'X-Kernel-Dashboard-Token':'existing-token'})
    assert result.status_code==200
    state=result.json()
    assert state['phase']=='Validation failed'
    assert state['protocol_digest']=='frozen-protocol'
    assert state['validation_attempts'][0]['state']=='failed'
    assert state['kernel_evals']==0 and '0 training episodes' in state['activity']
    assert 'unsubtracted launch overhead' in state['warning']
    assert state['cost'] is None
