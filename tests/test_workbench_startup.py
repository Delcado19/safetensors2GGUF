"""Guard browser readiness and fresh HTML without opening a real browser."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

import web_api


def test_browser_waits_until_http_ready(monkeypatch):
    """A refused connection must never open the browser before the next HTTP success."""
    fetch = Mock(side_effect=[OSError('not listening'), nullcontext(SimpleNamespace(status=200))])
    opened = Mock(return_value=True)
    sleep = Mock()
    monkeypatch.setattr('urllib.request.urlopen', fetch)
    monkeypatch.setattr('webbrowser.open', opened)
    monkeypatch.setattr(web_api.time, 'sleep', sleep)
    assert web_api.open_browser_when_ready('http://127.0.0.1:8789/')
    assert fetch.call_count == 2
    sleep.assert_called_once_with(0.2)
    opened.assert_called_once_with('http://127.0.0.1:8789/')


def test_failed_start_does_not_open_browser(monkeypatch):
    """Timeout leaves the browser closed when the server cannot serve the page."""
    opened = Mock()
    monkeypatch.setattr('webbrowser.open', opened)
    monkeypatch.setattr(web_api.time, 'monotonic', Mock(side_effect=[0, 2]))
    assert not web_api.open_browser_when_ready('http://127.0.0.1:8789/', timeout=1)
    opened.assert_not_called()


def test_html_is_not_cached_across_frontend_rebuilds(tmp_path, monkeypatch):
    """Fresh entry HTML must reference the assets shipped by the current build."""
    (tmp_path / 'index.html').write_text('<html>Workbench</html>')
    monkeypatch.setattr(web_api, 'DIST', tmp_path)
    response = TestClient(web_api.create_app(), base_url='http://localhost').get('/')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
