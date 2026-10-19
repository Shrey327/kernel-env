import json
from corrected_preflight import BoundedCases


CONTRACT={key:key for key in ('run_id','source_digest','protocol_digest','sft_sha256','teacher_dataset_sha256')}


def runner(tmp_path,names,clock=lambda:0,deadline=100):
    return BoundedCases(tmp_path,[{'case':name,'required':True} for name in names],deadline,tmp_path/'progress.json',clock)


def test_failed_case_keeps_later_success_and_complete_bound_summary(tmp_path):
    cases=runner(tmp_path,['early','later'])
    cases.run('early',lambda:({'observation':'failed sensitivity'},False))
    cases.run('later',lambda:({'observation':'independent success'},True))
    value=cases.finish(CONTRACT)
    assert value['passed'] is False and value['failures']==['early']
    assert cases.results['later']['passed']
    assert json.loads((tmp_path/'result.json').read_text())==value
    assert len((tmp_path/'cases.jsonl').read_text().splitlines())==2
    assert value['source_digest']==CONTRACT['source_digest']


def test_missing_case_and_deadline_are_explicit_notrun_and_fail_closed(tmp_path):
    cases=runner(tmp_path,['deadline','never reached'],clock=lambda:90)
    cases.run('deadline',lambda:(_ for _ in ()).throw(AssertionError('must not execute')))
    value=cases.finish(CONTRACT)
    assert not value['passed']
    assert cases.results['deadline']['status']=='notrun'
    assert 'deadline' in cases.results['deadline']['reason']
    assert cases.results['never reached']['status']=='notrun'
    assert len(value['failures'])==2


def test_identity_failure_blocks_dependents_but_independent_case_runs(tmp_path):
    cases=runner(tmp_path,['identity','dependent','independent'])
    cases.run('identity',lambda:({},False))
    cases.run('dependent',lambda:(_ for _ in ()).throw(AssertionError('unsafe')),requires=('identity',))
    cases.run('independent',lambda:({},True))
    assert cases.results['dependent']['status']=='notrun'
    assert cases.results['independent']['passed']
    assert not cases.finish(CONTRACT)['passed']


def test_gpu_health_failure_leaves_cpu_diagnostic_available(tmp_path):
    cases=runner(tmp_path,['broken gpu','later gpu','cpu'])
    def unhealthy():raise RuntimeError('device lost')
    cases.health=unhealthy
    cases.run('broken gpu',lambda:({},True),gpu=True)
    cases.run('later gpu',lambda:(_ for _ in ()).throw(AssertionError('unsafe')),gpu=True)
    cases.run('cpu',lambda:({},True))
    assert cases.results['later gpu']['status']=='notrun'
    assert 'device lost' in cases.results['later gpu']['reason']
    assert cases.results['cpu']['passed']


def test_optional_diagnostic_failure_does_not_clear_or_block_required_gate(tmp_path):
    cases=BoundedCases(tmp_path,[{'case':'small','required':False},{'case':'training','required':True}],100,tmp_path/'progress.json',lambda:0)
    cases.run('small',lambda:({},False));cases.run('training',lambda:({},True))
    assert cases.finish(CONTRACT)['passed']
