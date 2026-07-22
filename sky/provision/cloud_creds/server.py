"""Cloud credential registration API endpoints.

CRUD over per-name credential files on the API server (today: Nebius
service-account JSON files under ``~/.nebius/creds``). Mirrors the structure of
``sky/provision/slurm/server.py``: synchronous handlers that call the
registration core directly (no long-running work, so no ``executor``).

This is a control-plane surface that writes credential material with no
per-caller scoping, so the whole router is blocklisted for the ``user`` role in
``sky/users/rbac.py``.
"""
from typing import Any, Dict, List

import fastapi

from sky.provision.cloud_creds import registration
from sky.utils import common_utils

router = fastapi.APIRouter()


@router.get('')
def get_cloud_credentials() -> Dict[str, Dict[str, Any]]:
    """List registered cloud credentials (names + non-secret detail)."""
    try:
        return registration.list_creds()
    except Exception as e:  # pylint: disable=broad-except
        raise fastapi.HTTPException(
            status_code=500,
            detail='Failed to list cloud credentials: '
            f'{common_utils.format_exception(e)}')


@router.post('')
def register_cloud_credentials(cloud_cred: Dict[str, Any]) -> Dict[str, str]:
    """Upsert a cloud credential registration.

    Body fields: ``cloud`` (only ``nebius`` today), ``name``, and ``fields``
    -- for nebius: ``credentials_json`` (service-account credentials file
    contents), ``tenant_id``, and optionally ``domain``.
    """
    try:
        registration.register(
            cloud=cloud_cred['cloud'],
            name=cloud_cred['name'],
            fields=cloud_cred['fields'],
        )
        return {'status': 'success'}
    except KeyError as e:
        raise fastapi.HTTPException(status_code=400,
                                    detail=f'Missing required field: {e}')
    except ValueError as e:
        raise fastapi.HTTPException(status_code=400,
                                    detail=common_utils.format_exception(e))
    except Exception as e:  # pylint: disable=broad-except
        raise fastapi.HTTPException(
            status_code=500,
            detail='Failed to register cloud credentials: '
            f'{common_utils.format_exception(e)}')


@router.delete('/{name}')
def delete_cloud_credentials(name: str) -> Dict[str, str]:
    """Remove a cloud credential registration and its files."""
    try:
        if registration.delete(name):
            return {'status': 'success'}
        raise fastapi.HTTPException(
            status_code=404, detail=f'Cloud credential `{name}` not found')
    except fastapi.HTTPException:
        raise
    except ValueError as e:
        raise fastapi.HTTPException(status_code=400,
                                    detail=common_utils.format_exception(e))
    except Exception as e:  # pylint: disable=broad-except
        raise fastapi.HTTPException(
            status_code=500,
            detail='Failed to delete cloud credentials: '
            f'{common_utils.format_exception(e)}')


# Kept for symmetry with slurm_clusters; the router module is the single
# import surface for the registration API.
__all__: List[str] = ['router']
