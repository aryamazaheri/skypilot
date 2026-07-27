"""Constants for SSH Node Pools"""
# pylint: disable=line-too-long
import os

DEFAULT_KUBECONFIG_PATH = os.path.expanduser('~/.kube/config')
SSH_CONFIG_PATH = os.path.expanduser('~/.ssh/config')
NODE_POOLS_INFO_DIR = os.path.expanduser('~/.sky/ssh_node_pools_info')
NODE_POOLS_KEY_DIR = os.path.expanduser('~/.sky/ssh_keys')
DEFAULT_SSH_NODE_POOLS_PATH = os.path.expanduser('~/.sky/ssh_node_pools.yaml')

# Default sshd port for a pool host. A host (or a whole pool) may override it
# with a `port:` field, e.g. for a node reached through a reverse-SSH tunnel or
# a jump/NAT port mapping rather than directly on 22.
DEFAULT_SSH_PORT = 22

# TODO (kyuds): make this configurable?
K3S_TOKEN = 'mytoken'  # Any string can be used as the token
