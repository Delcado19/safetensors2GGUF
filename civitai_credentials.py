"""Civitai credentials: session override, environment, then Windows generic credential."""

import ctypes
from ctypes import wintypes
import os
import threading

VAULT_TARGET = 'safetensors2GGUF/Civitai'


class _Credential(ctypes.Structure):
    _fields_ = [('Flags', wintypes.DWORD), ('Type', wintypes.DWORD),
                ('TargetName', wintypes.LPWSTR), ('Comment', wintypes.LPWSTR),
                ('LastWritten', wintypes.FILETIME), ('CredentialBlobSize', wintypes.DWORD),
                ('CredentialBlob', ctypes.POINTER(ctypes.c_ubyte)),
                ('Persist', wintypes.DWORD), ('AttributeCount', wintypes.DWORD),
                ('Attributes', ctypes.c_void_p), ('TargetAlias', wintypes.LPWSTR),
                ('UserName', wintypes.LPWSTR)]


def _vault_key(value=None):
    """Read/write only our generic credential; never enumerate other stored secrets."""
    if os.name != 'nt':
        if value is not None:
            raise ValueError('Windows Credential Manager is available only on Windows.')
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
    if not api.CredReadW(VAULT_TARGET, 1, 0, ctypes.byref(pointer)):
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
        self._lock = threading.Lock()

    def resolve(self):
        """Return (key, source), prioritizing explicit session input over stored keys."""
        with self._lock:
            if self._session_key:
                return self._session_key, 'session'
        for name in ('CIVITAI_API_KEY', 'CIVITAI_TOKEN', 'CODEX_CIVITAI_API_KEY'):
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

    def set(self, key, persist=False):
        """Set a session override or clear it; persist only when explicitly selected."""
        key = key.strip()
        if len(key) > 1024 or any(ord(char) < 33 or ord(char) > 126 for char in key):
            raise ValueError('Enter a valid Civitai API key without spaces or control characters.')
        if persist:
            if not key:
                raise ValueError('Enter a key before saving to Windows Credential Manager.')
            _vault_key(key)
        with self._lock:
            self._session_key = key
        return self.status()
