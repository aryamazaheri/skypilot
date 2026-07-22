"""Remote registration of cloud credential files.

In client-server mode nothing can deliver *credential files* to the API server
remotely: workspace config can only *scope* clouds to files that already exist
(``nebius.credentials_file_path``, ``aws.profile``, ...), and the files those
settings point at live only on the API server's filesystem. This module
provides the core behind the ``/cloud_credentials`` CRUD router so an admin
control plane (e.g. a multi-tenant BYO-cloud enrollment flow) can deliver
per-name credential files over HTTP instead of writing daemon-local files by
hand.

Only ``cloud=nebius`` is supported today -- deliberately the first cloud
because its credential is one self-contained JSON file per name (no INI
profile merging like AWS, no kubeconfig context merging like Kubernetes).
Each registered name owns two files under ``~/.nebius/creds``:

    <name>.json        service-account credentials (0600); the target of a
                       workspace's ``nebius.credentials_file_path``;
    <name>.meta.json   non-secret sidecar ({cloud, tenant_id, domain}) so GET
                       can list registrations without touching key material.

This deliberately does NOT write the adaptor's global default paths
(``~/.nebius/credentials.json``, ``NEBIUS_TENANT_ID.txt``, ...): those are
process-global fallbacks, and a shared multi-tenant server must never have a
cloud identity that requests can fall back to. Workspace config carries the
``tenant_id``/``domain``/``credentials_file_path`` per tenant.

All writes are guarded by a distributed lock (``sky.utils.locks``) because
concurrent enrollments against a shared API server are the expected use case
(same discipline as ``sky.provision.slurm.registration``).
"""
import json
import os
import re
from typing import Any, Dict, Optional

from sky import sky_logging
from sky.utils import locks

logger = sky_logging.init_logger(__name__)

# Directory holding per-name Nebius credential files written by this API.
# Sibling of the adaptor's default ~/.nebius/credentials.json, but names are
# referenced explicitly via workspace `nebius.credentials_file_path` -- nothing
# reads this directory implicitly.
NEBIUS_CREDS_DIR = os.path.join('~', '.nebius', 'creds')

# Lock guarding all writes under the creds directories. Matches the filelock
# family used for workspace config updates; auto-detects postgres when
# configured.
_CLOUD_CREDS_LOCK_ID = 'cloud-creds-update'
_CLOUD_CREDS_LOCK_TIMEOUT_SECONDS = 60

# Clouds this endpoint can deliver credentials for. AWS/Kubernetes have
# fundamentally different write shapes (INI profile / kubeconfig merge) and are
# added here when their cases are built.
_SUPPORTED_CLOUDS = ('nebius',)

# A registered name is used as a filename (and lands in workspace YAML), so
# keep it to a safe charset. Same rule as slurm cluster registration.
_VALID_NAME_RE = re.compile(r'^[A-Za-z0-9._-]+$')

_META_SUFFIX = '.meta.json'


def _validate_name(name: str) -> None:
    if not name or not _VALID_NAME_RE.match(name):
        raise ValueError(
            f'Invalid cloud credential name {name!r}: names may only contain '
            'letters, digits, and the characters ".", "_", and "-".')


def _mask(value: str) -> str:
    """Mask an identifier, keeping only a short suffix so it stays
    recognizable without being disclosed in listings."""
    value = value.strip()
    if len(value) <= 4:
        return '****'
    return '****' + value[-4:]


class CloudCredsManager:
    """Write/read manager for per-name credential files on the API server."""

    def __init__(self) -> None:
        self.nebius_creds_dir = os.path.expanduser(NEBIUS_CREDS_DIR)

    # ---- public read API ------------------------------------------------

    def list_creds(self) -> Dict[str, Dict[str, Any]]:
        """Return registered credentials and their non-secret detail.

        Credential file *contents* are never returned -- only the sidecar
        metadata (with the tenant id masked) needed to identify an entry.
        """
        creds: Dict[str, Dict[str, Any]] = {}
        if not os.path.isdir(self.nebius_creds_dir):
            return creds
        for filename in sorted(os.listdir(self.nebius_creds_dir)):
            if not filename.endswith('.json') or filename.endswith(
                    _META_SUFFIX):
                continue
            name = filename[:-len('.json')]
            detail: Dict[str, Any] = {'cloud': 'nebius'}
            meta = self._read_meta(name)
            if meta.get('tenant_id'):
                detail['tenant_id'] = _mask(str(meta['tenant_id']))
            if meta.get('domain'):
                detail['domain'] = meta['domain']
            creds[name] = detail
        return creds

    # ---- public write API (lock-guarded) --------------------------------

    def register(self, cloud: str, name: str, fields: Dict[str, Any]) -> None:
        """Upsert the credential file(s) for ``name``.

        Args:
            cloud: Which cloud the credentials are for (only ``nebius``).
            name: Registration name, used as the credential filename.
            fields: Cloud-specific credential fields. For nebius:
                ``credentials_json`` (service-account credentials file
                contents, required), ``tenant_id`` (required), ``domain``
                (optional).
        """
        _validate_name(name)
        if cloud not in _SUPPORTED_CLOUDS:
            raise ValueError(
                f'Unsupported cloud {cloud!r} for credential registration '
                f'(supported: {", ".join(_SUPPORTED_CLOUDS)}).')
        credentials_json = fields.get('credentials_json')
        if not credentials_json:
            raise ValueError('`credentials_json` (file contents) is required.')
        try:
            json.loads(credentials_json)
        except (json.JSONDecodeError, TypeError) as e:
            # The message carries position info only, never file contents.
            raise ValueError(
                f'`credentials_json` is not valid JSON: {e}') from e
        tenant_id = fields.get('tenant_id')
        if not tenant_id:
            raise ValueError('`tenant_id` is required.')
        domain = fields.get('domain')

        meta = {'cloud': cloud, 'tenant_id': tenant_id}
        if domain:
            meta['domain'] = domain
        with locks.get_lock(_CLOUD_CREDS_LOCK_ID,
                            _CLOUD_CREDS_LOCK_TIMEOUT_SECONDS):
            self._write_secret(self._creds_path(name), credentials_json)
            self._write_secret(self._meta_path(name),
                               json.dumps(meta, sort_keys=True))

    def delete(self, name: str) -> bool:
        """Remove the credential file and its sidecar for ``name``.

        Returns True if a credential file was removed, False if none existed.
        """
        _validate_name(name)
        with locks.get_lock(_CLOUD_CREDS_LOCK_ID,
                            _CLOUD_CREDS_LOCK_TIMEOUT_SECONDS):
            existed = os.path.exists(self._creds_path(name))
            for path in (self._creds_path(name), self._meta_path(name)):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError as e:
                    logger.warning(f'Failed to remove {path}: {e}')
            return existed

    # ---- internals ------------------------------------------------------

    def _creds_path(self, name: str) -> str:
        return os.path.join(self.nebius_creds_dir, f'{name}.json')

    def _meta_path(self, name: str) -> str:
        return os.path.join(self.nebius_creds_dir, f'{name}{_META_SUFFIX}')

    def _read_meta(self, name: str) -> Dict[str, Any]:
        path = self._meta_path(name)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            return loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError) as e:
            logger.warning(f'Failed to read {path}: {e}')
            return {}

    @staticmethod
    def _write_secret(path: str, contents: str) -> None:
        """Atomically write ``contents`` 0600 (dir 0700): chmod the temp file
        BEFORE the rename so the final path is never readable, and replace so
        a concurrent reader never sees a half-written file."""
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        os.chmod(directory, 0o700)
        tmp_path = f'{path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(contents)
            if not contents.endswith('\n'):
                f.write('\n')
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)


# Module-level convenience wrappers (mirrors slurm.registration).


def list_creds() -> Dict[str, Dict[str, Any]]:
    return CloudCredsManager().list_creds()


def register(cloud: str, name: str, fields: Optional[Dict[str,
                                                          Any]]) -> None:
    CloudCredsManager().register(cloud=cloud, name=name, fields=fields or {})


def delete(name: str) -> bool:
    return CloudCredsManager().delete(name)
