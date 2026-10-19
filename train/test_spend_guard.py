import json

import pytest

from spend_guard import SpendGuard


def ledger(tmp_path):
    path=tmp_path/'budget.json'
    path.write_text(json.dumps({'new_jobs':[],'additional_cap_usd':100,
        'envelopes_usd':{'validation_warmup_pilot':15,'training':60,'evaluation':15}}))
    return SpendGuard(path)


def test_active_or_uncertain_launch_keeps_its_entire_reservation(tmp_path):
    guard=ledger(tmp_path)
    row=guard.reserve('validate','validation_warmup_pilot',1200)
    assert row['reserved_upper_usd']==pytest.approx(10/3)
    with pytest.raises(RuntimeError):guard.reserve('another','training',100)
    guard.record(row['reservation_id'],state='failed')
    assert guard.reserve('retry','validation_warmup_pilot',1200)['reserved_upper_usd']==pytest.approx(10/3)


def test_all_retries_share_cap_and_envelopes(tmp_path):
    guard=ledger(tmp_path)
    for i in range(4):
        row=guard.reserve(str(i),'validation_warmup_pilot',1200)
        guard.record(row['reservation_id'],state='failed')
    with pytest.raises(RuntimeError):guard.reserve('fifth','validation_warmup_pilot',1200)
    row=guard.reserve('train','training',6*3600)
    guard.record(row['reservation_id'],state='complete')
    with pytest.raises(RuntimeError):guard.reserve('train-again','training',1)
