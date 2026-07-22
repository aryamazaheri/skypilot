"""Tests for the remote cloud credential registration core."""
import json
import os
import stat

import pytest

from sky.provision.cloud_creds import registration

_CREDS_JSON = json.dumps({
    'subject-credentials': {
        'type': 'JWT',
        'kid': 'publickey-e00SECRET',
        'iss': 'serviceaccount-e00abc',
    }
})


@pytest.fixture
def manager(tmp_path):
    """A CloudCredsManager pointed at an isolated tmp directory."""
    m = registration.CloudCredsManager()
    m.nebius_creds_dir = str(tmp_path / 'creds')
    return m


def _register(manager, name='t-abc-neb', **overrides):
    fields = {
        'credentials_json': _CREDS_JSON,
        'tenant_id': 'tenant-e00faketen1234',
    }
    fields.update(overrides)
    manager.register('nebius', name, fields)


class TestRegister:

    def test_register_writes_0600_creds_file(self, manager):
        _register(manager)
        creds_path = os.path.join(manager.nebius_creds_dir, 't-abc-neb.json')
        assert os.path.exists(creds_path)
        assert stat.S_IMODE(os.stat(creds_path).st_mode) == 0o600
        assert json.load(open(creds_path)) == json.loads(_CREDS_JSON)
        # The containing directory is private too.
        assert stat.S_IMODE(os.stat(
            manager.nebius_creds_dir).st_mode) == 0o700

    def test_register_writes_non_secret_sidecar(self, manager):
        _register(manager, domain='api.custom.nebius.cloud')
        meta_path = os.path.join(manager.nebius_creds_dir,
                                 't-abc-neb.meta.json')
        meta = json.load(open(meta_path))
        assert meta == {
            'cloud': 'nebius',
            'tenant_id': 'tenant-e00faketen1234',
            'domain': 'api.custom.nebius.cloud',
        }
        # The sidecar never holds key material.
        assert 'SECRET' not in open(meta_path).read()

    def test_upsert_replaces_in_place(self, manager):
        _register(manager)
        new_creds = json.dumps({'subject-credentials': {'kid': 'rotated'}})
        _register(manager,
                  credentials_json=new_creds,
                  tenant_id='tenant-e00other0000')
        creds_path = os.path.join(manager.nebius_creds_dir, 't-abc-neb.json')
        assert json.load(open(creds_path)) == json.loads(new_creds)
        assert len(manager.list_creds()) == 1

    def test_unsupported_cloud_rejected(self, manager):
        with pytest.raises(ValueError, match='Unsupported cloud'):
            manager.register('aws', 't-abc-aws', {'credentials_json': '{}'})

    @pytest.mark.parametrize('bad_name', ['../evil', 'a/b', '', 'a b'])
    def test_invalid_names_rejected(self, manager, bad_name):
        with pytest.raises(ValueError):
            _register(manager, name=bad_name)

    def test_non_json_credentials_rejected(self, manager):
        with pytest.raises(ValueError, match='not valid JSON'):
            _register(manager, credentials_json='not json at all {')
        # Nothing was written.
        assert manager.list_creds() == {}

    @pytest.mark.parametrize('missing', ['credentials_json', 'tenant_id'])
    def test_required_fields(self, manager, missing):
        with pytest.raises(ValueError):
            _register(manager, **{missing: ''})


class TestDelete:

    def test_delete_removes_both_files(self, manager):
        _register(manager)
        creds_path = os.path.join(manager.nebius_creds_dir, 't-abc-neb.json')
        meta_path = os.path.join(manager.nebius_creds_dir,
                                 't-abc-neb.meta.json')
        assert os.path.exists(creds_path) and os.path.exists(meta_path)

        assert manager.delete('t-abc-neb') is True
        assert not os.path.exists(creds_path)
        assert not os.path.exists(meta_path)

    def test_delete_missing_returns_false(self, manager):
        _register(manager)
        assert manager.delete('t-other-neb') is False
        # The existing registration is untouched.
        assert set(manager.list_creds()) == {'t-abc-neb'}

    def test_delete_leaves_other_registrations(self, manager):
        _register(manager, name='t-abc-neb')
        _register(manager, name='t-def-neb')
        manager.delete('t-abc-neb')
        assert set(manager.list_creds()) == {'t-def-neb'}


class TestListCreds:

    def test_list_empty_when_no_dir(self, manager):
        assert manager.list_creds() == {}

    def test_list_returns_masked_non_secret_detail(self, manager):
        _register(manager, domain='api.custom.nebius.cloud')
        creds = manager.list_creds()
        assert set(creds) == {'t-abc-neb'}
        entry = creds['t-abc-neb']
        assert entry['cloud'] == 'nebius'
        # Tenant id is masked to a short suffix.
        assert entry['tenant_id'] == '****1234'
        assert entry['domain'] == 'api.custom.nebius.cloud'
        # Credential contents are never exposed via listing.
        assert 'SECRET' not in str(creds)

    def test_list_skips_domain_when_absent(self, manager):
        _register(manager)
        assert 'domain' not in manager.list_creds()['t-abc-neb']

    def test_list_survives_missing_sidecar(self, manager):
        _register(manager)
        os.remove(
            os.path.join(manager.nebius_creds_dir, 't-abc-neb.meta.json'))
        assert manager.list_creds() == {'t-abc-neb': {'cloud': 'nebius'}}
