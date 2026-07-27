"""Per-host `port` support for SSH Node Pools.

A pool host may be reached on a non-default sshd port (e.g. a node behind NAT
whose sshd is exposed as a reverse-SSH forward on a bastion). These tests pin
the three places that must honour it: host-info resolution, the ssh command
builder every deploy step routes through, and the kubectl tunnel exec-args.
"""
import os
from unittest import mock

import pytest

from sky.ssh_node_pools import utils as ssh_utils
from sky.ssh_node_pools.deploy import tunnel_utils
from sky.ssh_node_pools.deploy import utils as deploy_utils


def _hosts(cluster_config):
    with mock.patch.object(ssh_utils, 'check_host_in_ssh_config',
                           return_value=False):
        return ssh_utils.prepare_hosts_info('pool', cluster_config)


def test_prepare_hosts_info_defaults_to_22():
    info = _hosts({'hosts': ['1.2.3.4'], 'user': 'ubuntu'})
    assert info[0]['port'] == 22


def test_prepare_hosts_info_per_host_and_cluster_port():
    info = _hosts({
        'hosts': ['1.2.3.4', {
            'ip': '5.6.7.8',
            'port': 20001
        }],
        'user': 'ubuntu',
        'port': 2222,
    })
    # Cluster-level port is inherited; a host-level port wins.
    assert info[0]['port'] == 2222
    assert info[1]['port'] == 20001


def test_prepare_hosts_info_port_from_yaml_string():
    info = _hosts({'hosts': [{'ip': '1.2.3.4', 'port': '20001'}],
                   'user': 'ubuntu'})
    assert info[0]['port'] == 20001


def _argv(**kwargs):
    with mock.patch('subprocess.run') as run:
        run.return_value = mock.Mock(returncode=0, stdout='', stderr='')
        deploy_utils.run_remote('host', 'echo hi', 'ubuntu', **kwargs)
        return run.call_args[0][0]


def test_run_remote_adds_port_flag():
    assert ['-p', '20001'] == _argv(port=20001)[-4:-2]


def test_run_remote_omits_default_port():
    assert '-p' not in _argv()
    assert '-p' not in _argv(port=22)


def test_run_remote_ssh_config_ignores_port():
    # The user's ssh config supplies its own Port; we must not override it.
    assert _argv(use_ssh_config=True, port=20001) == ['ssh', 'host', 'echo hi']


def test_tunnel_exec_args_carry_ssh_port():
    with mock.patch.object(tunnel_utils.deploy_utils, 'run_command') as run, \
            mock.patch.object(tunnel_utils, 'get_available_port',
                              return_value=6443), \
            mock.patch.object(os.path, 'isfile', return_value=False), \
            mock.patch.object(os, 'chmod'):
        tunnel_utils.setup_kubectl_ssh_tunnel('1.2.3.4',
                                              'ubuntu',
                                              '/tmp/key',
                                              'ssh-pool',
                                              ssh_port=20001)
    args = [c[0][0] for c in run.call_args_list]
    creds = [a for a in args if 'set-credentials' in a][0]
    assert '--exec-arg=--ssh-port' in creds
    assert '--exec-arg=20001' in creds


def test_ssh_config_host_with_explicit_port_is_refused():
    """An ssh-config host takes its Port from that config, so an explicit `port` here
    would be silently ignored and the deploy would dial elsewhere. Ambiguous inputs
    must fail loudly — the bastion address this feature uses is exactly the kind of
    name people put in ~/.ssh/config."""
    with mock.patch.object(ssh_utils, 'check_host_in_ssh_config',
                           return_value=True):
        with pytest.raises(ValueError, match='SSH config'):
            ssh_utils.prepare_hosts_info(
                'pool', {'hosts': [{'ip': 'bastion', 'port': 20001}],
                         'user': 'ubuntu'})
        # port 22 is what ssh assumes anyway -> no conflict, no error.
        info = ssh_utils.prepare_hosts_info(
            'pool', {'hosts': ['bastion'], 'user': 'ubuntu'})
        assert info[0]['use_ssh_config'] is True


def test_scp_of_kubeconfig_uses_capital_P_for_the_port():
    """scp spells the port -P, not -p; this is the one call site where the flag letter
    differs from ssh, so it is the easiest to silently regress."""
    import inspect

    from sky.ssh_node_pools.deploy import deploy

    lines = inspect.getsource(deploy.deploy_single_cluster).splitlines()
    start = next(i for i, ln in enumerate(lines) if 'scp_cmd = [' in ln)
    end = next(i for i, ln in enumerate(lines) if 'run_command(scp_cmd' in ln)
    block = '\n'.join(lines[start:end])
    assert "'-P'" in block and 'head_port' in block
    assert "'-p'" not in block


def _k3s_install_block() -> str:
    """The head-node k3s install command, as a single source block."""
    import inspect

    from sky.ssh_node_pools.deploy import deploy

    lines = inspect.getsource(deploy.deploy_single_cluster).splitlines()
    start = next(i for i, ln in enumerate(lines) if 'get.k3s.io' in ln)
    end = next(i for i, ln in enumerate(lines[start:], start)
               if 'run_remote' in ln)
    return '\n'.join(lines[start:end])


def test_head_node_wait_targets_that_node_not_every_node():
    """`kubectl wait ... node --all` waits for EVERY node in the cluster, so one
    permanently-NotReady leftover (e.g. a node object from an earlier enrollment of
    the same box, named by its hostname) makes every future deploy of that machine
    time out and fail — observed live: 3x2min then "Failed to deploy K3s". Wait for
    the node this install just created instead."""
    block = _k3s_install_block()
    assert 'node/{head_node}' in block
    assert '--all' not in block


def test_head_node_wait_does_not_fail_on_a_successful_third_attempt():
    """The old loop tested `[ $i -eq 3 ]` afterwards, so succeeding on the LAST
    attempt was reported as failure (i is still 3 after `break`). Success must be
    tracked explicitly, not inferred from the counter."""
    block = _k3s_install_block()
    assert 'ready=1' in block
    assert '$i -eq 3' not in block
