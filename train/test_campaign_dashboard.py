import json
from fastapi.testclient import TestClient
from campaign_dashboard import make_app, state_from


def test_dashboard_is_authenticated_read_only_and_never_accepts_paths(tmp_path):
    root=tmp_path/'experiment';root.mkdir()
    (root/'manifest.json').write_text(json.dumps({'model':'test','steps':64,'episodes_per_update':32,'budget_cap_usd':100}))
    folder=root/'seed42/adapter';folder.mkdir(parents=True)
    (folder/'live_trainer_state.json').write_text(json.dumps({'global_step':3,'log_history':[{'step':3,'loss':.01}]}))
    with TestClient(make_app(root,'test-secret')) as client:
        assert client.get('/health').status_code==200
        assert client.get('/state').status_code==401
        headers={'X-Kernel-Dashboard-Token':'test-secret'}
        state=client.get('/state',headers=headers).json()
        assert state['completed']==3 and state['budget_cap_usd']==100 and state['episodes_per_update']==32
        assert client.get('/state?run=../../other',headers=headers).status_code==404
        assert client.post('/state',headers=headers).status_code==405
        assert client.get('/rollouts.jsonl',headers=headers).status_code==404
        assert client.get('/metrics.json',headers=headers).json()['3']['loss']==.01


def test_telemetry_counts_submissions_before_episodes_finish_and_includes_startup(tmp_path):
    root=tmp_path/'experiment';root.mkdir()
    (root/'manifest.json').write_text(json.dumps({'created_at_unix':1500,'deadline_unix':1000+8*3600-120,'prior_reported_usd':5}))
    (root/'status.json').write_text(json.dumps({'phase':'Stopped for reward-noise fix','updated_unix':4600}))
    (tmp_path/'evals.jsonl').write_text('\n'.join(json.dumps({'event':'step','queue_wait_s':4,'eval_s':2}) for _ in range(510)))
    state=state_from(root,'seed42')
    assert state['kernel_evals']==510 and state['recent_evaluation']['calls']==500
    assert state['recent_evaluation']['queue_wait_s']==4
    assert state['estimated_upper_usd']==14.7


def test_serves_ui_shells_publicly_and_unlocks_data_with_cookie(tmp_path):
    root=tmp_path/'experiment';(root/'seed42').mkdir(parents=True)
    with TestClient(make_app(root,'test-secret'),base_url='https://testserver') as client:
        for path in ('/','/methodology','/journey.html','/pilot/','/pilot/journey.html'):
            response=client.get(path)
            assert response.status_code==200 and 'text/html' in response.headers['content-type'],path
        assert 'Kernel Lab' in client.get('/').text
        for path in ('/state','/pilot/state.json','/pilot/../campaign_dashboard.py','/auth'):
            assert client.get(path).status_code in (401,404),path
        assert client.get('/auth',headers={'X-Kernel-Dashboard-Token':'wrong'}).status_code==401
        login=client.get('/auth',headers={'X-Kernel-Dashboard-Token':'test-secret'})
        assert login.status_code==204 and 'httponly' in login.headers['set-cookie'].lower()
        assert client.get('/state').status_code==200
        assert client.get('/pilot/state.json').status_code==200
        assert client.get('/pilot/missing.json').status_code==404
