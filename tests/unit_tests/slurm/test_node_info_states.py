"""Tests for sinfo node-state handling in _get_slurm_node_info_list.

sinfo's %t short state carries single-char flag suffixes ('-' planned,
'*' not responding, '$' maintenance, ...). Free-GPU accounting must compare
the BASE state: a MIXED+PLANNED node reports 'mix-', and an exact-match test
used to skip the allocation subtraction — reporting a fully busy node as
fully free (seen live: 40/40 A100 "free" on a cluster whose A100 nodes all
sat in 'mix-').
"""

from unittest import mock

import pytest

from sky.adaptors.slurm import NodeInfo
from sky.provision.slurm.utils import _get_slurm_node_info_list


def _node(name: str, state: str, gres: str = 'gpu:a100:8') -> NodeInfo:
    return NodeInfo(node=name,
                    state=state,
                    gres=gres,
                    cpus=64,
                    memory_gb=512.0,
                    partition='acc')


class _FakeSlurmClient:

    def __init__(self, nodes, jobs_gres):
        self._nodes = nodes
        self._jobs_gres = jobs_gres

    def info_nodes(self):
        return self._nodes

    def get_all_jobs_gres(self):
        return self._jobs_gres


def _node_info_list(nodes, jobs_gres):
    fake_config = mock.Mock()
    fake_config.lookup.return_value = {
        'hostname': 'login.example',
        'user': 'user',
        'identityfile': ['/tmp/key'],
    }
    with mock.patch(
            'sky.provision.slurm.utils.get_slurm_ssh_config',
            return_value=fake_config), \
         mock.patch('sky.provision.slurm.utils.slurm.SlurmClient',
                    return_value=_FakeSlurmClient(nodes, jobs_gres)):
        return _get_slurm_node_info_list(slurm_cluster_name='cluster1')


@pytest.mark.parametrize(
    'state,jobs_gres,expected_free',
    [
        # Base states (no suffix) — unchanged behavior.
        ('idle', {}, 8),
        ('mix', {'n1': ['gpu:a100:6']}, 2),
        ('alloc', {}, 0),           # alloc with no GRES info -> assume all used
        ('down', {}, 0),
        # Flagged states — the bug: these used to read as fully free.
        ('mix-', {'n1': ['gpu:a100:6']}, 2),      # MIXED + PLANNED
        ('mix-', {'n1': ['gpu:a100:8']}, 0),      # fully allocated + planned
        ('alloc*', {}, 0),                        # allocated + not responding
        ('drain$', {'n1': ['gpu:a100:2']}, 0),    # draining + maintenance
        ('down*', {}, 0),                         # down + not responding
        ('idle~', {}, 8),                         # powered-down idle stays free
        # Multiple jobs on one node sum up.
        ('mix-', {'n1': ['gpu:a100:3', 'gpu:a100:4']}, 1),
    ])
def test_free_gpus_respects_state_flag_suffixes(state, jobs_gres,
                                                expected_free):
    infos = _node_info_list([_node('n1', state)], jobs_gres)
    assert len(infos) == 1
    assert infos[0]['total_gpus'] == 8
    assert infos[0]['free_gpus'] == expected_free
