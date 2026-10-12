"""Civitai credentials: session override, environment, then Windows generic credential."""

import ctypes
from ctypes import wintypes
import os
import threading

VAULT_TARGET = 'safetensors2GGUF/Civitai'
ENV_NAMES = ('CIVITAI_API_KEY', 'CIVITAI_TOKEN', 'CODEX_CIVITAI_API_KEY')


class CredentialError(ValueError):
    """Safe credential setup errors with fixed text and no user input or secrets."""


class _Credential(ctypes.Structure):
    _fields_ = [('Flags', wintypes.DWORD), ('Type', wintypes.DWORD),
                ('TargetName', wintypes.LPWSTR), ('Comment', wintypes.LPWSTR),
                ('LastWritten', wintypes.FILETIME), ('CredentialBlobSize', wintypes.DWORD),
                ('CredentialBlob', ctypes.POINTER(ctypes.c_ubyte)),
                ('Persist', wintypes.DWORD), ('AttributeCount', wintypes.DWORD),
                ('Attributes', ctypes.c_void_p), ('TargetAlias', wintypes.LPWSTR),
                ('UserName', wintypes.LPWSTR)]


def _vault_key(value=None, target=VAULT_TARGET):
    """Read a named generic credential or write our default; never enumerate secrets."""
    if os.name != 'nt':
        if value is not None:
            raise CredentialError('Windows Credential Manager is available only on Windows.')
        return ''
    api = ctypes.WinDLL('Advapi32.dll', use_last_error=True)
    if value is not None:
        blob = value.encode('utf-16-le')
        buffer = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        credential = _Credential(Type=1, TargetName=VAULT_TARGET,
                                 CredentialBlobSize=len(blob), CredentialBlob=buffer,
                                 Persist=2, UserName='Civitai')
        api.CredWriteW.argtypes = [ctypes.POINTER(_Credential), wintypes.DWORD]
        if not api.CredWriteW(ctypes.byref(credential), 0):
            raise OSError('Could not save the Civitai key in Windows Credential Manager.')
        return value
    pointer = ctypes.POINTER(_Credential)()
    api.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                             ctypes.POINTER(ctypes.POINTER(_Credential))]
    api.CredFree.argtypes = [ctypes.c_void_p]
    if not api.CredReadW(target, 1, 0, ctypes.byref(pointer)):
        if ctypes.get_last_error() == 1168:  # ERROR_NOT_FOUND: no configured key.
            return ''
        raise OSError('Could not read the Civitai key from Windows Credential Manager.')
    try:
        credential = pointer.contents
        return ctypes.string_at(credential.CredentialBlob,
                                credential.CredentialBlobSize).decode('utf-16-le')
    finally:
        api.CredFree(pointer)


class CivitaiCredentials:
    """Keep manual keys process-local; expose status without returning secret values."""

    def __init__(self):
        self._session_key = ''
        self._session_source = 'session'
        self._lock = threading.Lock()

    def resolve(self):
        """Return (key, source), prioritizing explicit session input over stored keys."""
        with self._lock:
            if self._session_key:
                return self._session_key, self._session_source
        for name in ENV_NAMES:
            value = os.environ.get(name, '').strip()
            if value:
                return value, 'environment'
        value = _vault_key()
        return value, 'windows_vault' if value else 'none'

    def status(self):
        """Report credential presence and source, never the key itself."""
        try:
            key, source = self.resolve()
            return {'configured': bool(key), 'source': source, 'vault_available': os.name == 'nt'}
        except (OSError, UnicodeError):
            return {'configured': False, 'source': 'unavailable', 'vault_available': os.name == 'nt'}

    def set(self, key, persist=False, source='direct'):
        """Resolve an explicit key/env/vault source into a session override atomically."""
        key = key.strip()
        if source not in {'direct', 'environment', 'windows_vault'}:
            raise CredentialError('Choose a credential source.')
        if source != 'direct' and (not key or '\0' in key or len(key) > 1024 or persist):
            raise CredentialError('Enter a credential name; persistence is only available for direct keys.')
        # Names are resolved only in explicit reference modes; direct input is never guessed.
        if source == 'environment':
            if '=' in key:
                raise CredentialError('Enter a valid environment variable name.')
            key = os.environ.get(key, '').strip()
            if not key:
                raise CredentialError('The selected environment variable is not set in the server process. Restart the Workbench after changing environment variables.')
        elif source == 'windows_vault':
            key = _vault_key(target=key).strip()
            if not key:
                raise CredentialError('The selected Windows credential was not found.')
        if len(key) > 1024 or any(ord(char) < 33 or ord(char) > 126 for char in key):
            raise CredentialError('Enter a valid Civitai API key without spaces or control characters.')
        if persist:
            if not key:
                raise CredentialError('Enter a key before saving to Windows Credential Manager.')
            _vault_key(key)
        with self._lock:
            self._session_key = key
            self._session_source = 'session' if source == 'direct' else f'session_{source}'
        return self.status()
