import importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
spec=importlib.util.spec_from_file_location('corrected_launcher',Path(__file__).parents[1]/'launch_corrected.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_detached_exception_and_timeout_cancel_and_stop_only_owned_app():
    events=[]
    call=SimpleNamespace(cancel=lambda **kwargs:events.append(('cancel',kwargs)))
    proof=m.cleanup('ap-owned',call,False,stop_app=lambda:events.append('stop'),is_stopped=lambda app:app=='ap-owned')
    assert proof['zero_workers_verified'] and events==[('cancel',{'terminate_containers':True}),'stop']
    events.clear();m.cleanup('ap-owned',None,False,stop_app=lambda:events.append('stop'),is_stopped=lambda app:True)
    assert events==['stop']


def test_success_also_stops_resource_and_unverified_shutdown_fails_closed(monkeypatch):
    events=[]
    m.cleanup('ap-owned',None,True,stop_app=lambda:events.append('stop'),is_stopped=lambda app:True)
    assert events==['stop']
    clock=iter([0,21]);monkeypatch.setattr(m.time,'monotonic',lambda:next(clock))
    with pytest.raises(RuntimeError,match='zero workers unconfirmed'):
        m.cleanup('ap-owned',None,True,stop_app=lambda:None,is_stopped=lambda app:False)


def test_train_cannot_launch_without_matching_independent_review():
    from corrected_contract import RUN
    c={k:'x' for k in ('source_digest','protocol_digest','sft_sha256','teacher_dataset_sha256')};c['run_id']=RUN
    local={**c,'passed':True,'stage':'local'}
    with pytest.raises(ValueError,match='gpu-validation'):m.check_launch('train',c,local,{})
    m.check_launch('validate',c,local,{})


def test_marker_read_timeout_is_bounded(monkeypatch):
    def timeout(*args,**kwargs):
        assert kwargs['timeout']==10
        raise m.subprocess.TimeoutExpired('readonly progress',10)
    monkeypatch.setattr(m.subprocess,'run',timeout)
    with pytest.raises(m.subprocess.TimeoutExpired):m.progress('owned-reservation')


def test_hanging_cancel_cannot_prevent_app_stop():
    import threading
    release=threading.Event();events=[]
    def hung(**kwargs):release.wait(5)
    try:
        result=m.cleanup('ap-owned',SimpleNamespace(cancel=hung),False,stop_app=lambda:events.append('stopped'),
                         is_stopped=lambda app:True,cancel_timeout=.01)
        assert events==['stopped'] and any('cancel_timeout' in x for x in result['cancel_errors'])
    finally:release.set()


def test_eval_no_progress_bound_covers_600s_worker_allowance():
    # Evaluation workers get 600s per eval (modal_corrected); the launcher must not stop them first.
    assert all(m.no_progress_seconds(p)==660 for p in ('validate','eval-base','eval-sft','eval-rl'))
    assert m.no_progress_seconds('pilot')==m.no_progress_seconds('train')==300
