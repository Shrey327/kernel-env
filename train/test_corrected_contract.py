import json
import pytest
from corrected_contract import require_gate,digest,RUN,verify_sources
from spend_guard import SpendGuard


def test_gate_is_bound_to_every_protocol_source_and_sft_identity():
    c={k:'x' for k in ('run_id','source_digest','protocol_digest','sft_sha256','teacher_dataset_sha256')};c['run_id']=RUN
    gate={**c,'stage':'pilot','passed':True}
    require_gate(gate,c,'pilot')
    for key in c:
        with pytest.raises(ValueError,match='mismatch'):require_gate({**gate,key:'stale'},c,'pilot')
    with pytest.raises(ValueError):require_gate({**gate,'passed':False},c,'pilot')


def test_source_change_invalidates_gate(tmp_path):
    (tmp_path/'worker.py').write_text('old')
    expected={'worker.py':__import__('hashlib').sha256(b'old').hexdigest()}
    verify_sources(tmp_path,expected)
    (tmp_path/'worker.py').write_text('new')
    with pytest.raises(ValueError):verify_sources(tmp_path,expected)


def test_evaluation_funds_cannot_be_consumed_by_training_or_retries(tmp_path):
    p=tmp_path/'new.json';p.write_text(json.dumps({'new_jobs':[],'additional_cap_usd':100,'auxiliary_and_contingency_reserve_usd':5,
       'protected_evaluation_usd':30,'envelopes_usd':{'training':100,'evaluation':30}}));g=SpendGuard(p)
    first=g.reserve('train','training',6*3600);g.record(first['reservation_id'],state='failed')
    with pytest.raises(RuntimeError):g.reserve('retry','training',3600)
    row=g.reserve('evaluation','evaluation',3600);assert row['reserved_upper_usd']==10
    assert json.loads(p.read_text())['auxiliary_and_contingency_reserve_usd']==5


def test_recovery_protocol_pins_shapes_sft_identity_and_unsubtracted_timer():
    import os
    from corrected_contract import protocol
    from better_protocol import TRAINING_SHAPES,TRAINING_VARIANT
    from kernel_env.server.cuda_graph_timing import METHOD
    if not os.environ.get('KERNELBENCH_DIR'):pytest.skip('pinned KernelBench source required')
    value=protocol({'adapter_sha256':'saved-adapter','examples_sha256':'saved-original-dataset'})
    assert value['timing_method']==METHOD=='fresh_paired_cuda_graph_event_interval_v5'
    assert 'unsubtracted' in value['measurement'] and 'Not pure kernel time' in value['measurement']
    assert value['training_shapes']==TRAINING_SHAPES and value['training_variant']==TRAINING_VARIANT==0
    assert value['sft_sha256']=='saved-adapter' and value['teacher_dataset_sha256']=='saved-original-dataset'
    assert value['authorization_boundary_utc']=='2026-10-19T16:01:11Z' and value['fresh_cap_usd']==100
    assert 'Not pure kernel time' in value['measurement']
