import asyncio

import pytest

import kernel_env.client as module
from kernel_env.client import EnvironmentInfrastructureError, KernelEnv


def test_transient_handshakes_retry_with_bounded_backoff(monkeypatch):
    calls, sleeps = [], []
    connection = object()

    async def connect(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise TimeoutError('opening handshake')
        return connection

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(module, 'ws_connect', connect)
    monkeypatch.setattr(module.asyncio, 'sleep', sleep)
    client = KernelEnv(base_url='http://127.0.0.1:8000', connect_timeout_s=2)
    assert asyncio.run(client.connect()) is client
    assert client._ws is connection and len(calls) == 3
    assert sleeps == [0.25, 0.5]
    assert all(c['open_timeout'] == 2 for c in calls)
    client._ws = None


@pytest.mark.parametrize('error,expected_calls', [(TimeoutError('timeout'), 4), (ValueError('bad URL'), 1)])
def test_permanent_and_exhausted_errors_are_infrastructure(monkeypatch, error, expected_calls):
    calls = []

    async def connect(*args, **kwargs):
        calls.append(1)
        raise error

    async def sleep(seconds):
        pass

    monkeypatch.setattr(module, 'ws_connect', connect)
    monkeypatch.setattr(module.asyncio, 'sleep', sleep)
    with pytest.raises(EnvironmentInfrastructureError):
        asyncio.run(KernelEnv(base_url='http://127.0.0.1:8000').connect())
    assert len(calls) == expected_calls


def test_infrastructure_result_is_never_a_zero_reward_policy_sample():
    client = KernelEnv(base_url='http://127.0.0.1:8000')
    with pytest.raises(EnvironmentInfrastructureError, match='verifier OOM'):
        client._parse_result({'observation': {'stage':'infrastructure', 'feedback':'verifier OOM'}, 'reward':0})
