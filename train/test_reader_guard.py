import json
from types import SimpleNamespace
import pytest
import reader_guard as guard


def policy():
    return {'app_id':guard.APP_ID,'service_name':guard.SERVICE,'expires_unix':guard.EXPIRES_UNIX,
            'created_unix':guard.EXPIRES_UNIX-21600,'max_cost_usd':.5}


def test_fixed_original_identity_and_expiry_are_mandatory():
    guard.validate_policy(policy())
    for key,value in [('app_id','ap-other'),('expires_unix',guard.EXPIRES_UNIX+1),
                      ('created_unix',guard.EXPIRES_UNIX-3600),('max_cost_usd',1)]:
        with pytest.raises(ValueError):guard.validate_policy({**policy(),key:value})
    assert guard.stop_reason(guard.CUTOFF_UNIX-1) is None
    assert guard.stop_reason(guard.CUTOFF_UNIX)=='original lease cutoff'
    assert guard.stop_reason(guard.CUTOFF_UNIX-1,True)=='experiment terminal'
    assert set(guard.MARKERS)=={run+'/'+name for run in ('better-20261019','corrected-20261019')
                              for name in ('comparison.json','TERMINAL.json')}


def test_stop_success_requires_exact_app_stopped_and_zero_workers():
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=json.dumps([{'App ID':guard.APP_ID,'State':'stopped','Tasks':0}]))
    result=guard.stop_and_verify(run=run)
    assert result['zero_workers_verified']
    assert calls==[['modal','app','stop',guard.APP_ID],['modal','app','list','--json']]
    assert not guard.stopped([{'App ID':'ap-other','State':'stopped','Tasks':0}])
    assert not guard.stopped([{'App ID':guard.APP_ID,'State':'stopped','Tasks':1}])


def test_no_zero_worker_proof_is_uncertain_after_bounded_wait():
    now=[0.]
    def run(cmd,**kwargs):
        assert kwargs['timeout']<=12
        return SimpleNamespace(stdout='[]')
    def sleep(seconds):now[0]+=seconds
    result=guard.stop_and_verify(run=run,monotonic=lambda:now[0],sleep=sleep)
    assert now[0]==28
    assert not result['zero_workers_verified']
    assert result['outcome'].startswith('uncertain:')
