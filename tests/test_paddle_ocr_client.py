"""Regression test for the async jobs flow of :class:`PaddleOCRClient`.

The client submits a file, polls the job to ``done``, downloads the result
JSONL and materializes its pages. This pins that contract against a fake
session so a future API drift fails loudly without hitting the network.
"""
import json

from ingestion.paddle_ocr.client import PaddleOCRClient

_API = "https://fake.example/api/v2/ocr/jobs"
_JSONL_URL = "https://cdn.example/result.jsonl"
_IMG_URL = "https://cdn.example/img0.png"
_OUT_URL = "https://cdn.example/layout.png"
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


class _Resp:
    def __init__(self, *, json_body=None, text=None, content=None):
        self._json = json_body
        self.text = text if text is not None else json.dumps(json_body or {})
        self.content = content if content is not None else b""
        self.status_code = 200

    def json(self):
        return self._json

    def raise_for_status(self):
        return None


class _FakeSession:
    """Routes the submit POST and the status/jsonl/image GETs by url."""

    def __init__(self):
        self.posted = None
        self._status_calls = 0

    def post(self, url, headers=None, data=None, files=None, timeout=None, **kw):
        self.posted = {"url": url, "headers": headers, "data": data, "files": files}
        return _Resp(json_body={"data": {"jobId": "job-123"}})

    def get(self, url, headers=None, timeout=None, **kw):
        if url == f"{_API}/job-123":
            self._status_calls += 1
            if self._status_calls == 1:
                return _Resp(
                    json_body={"data": {"state": "running",
                                        "extractProgress": {"extractedPages": 1,
                                                            "totalPages": 2}}}
                )
            return _Resp(
                json_body={"data": {"state": "done",
                                    "resultUrl": {"jsonUrl": _JSONL_URL},
                                    "extractProgress": {"extractedPages": 2,
                                                        "totalPages": 2}}}
            )
        if url == _JSONL_URL:
            lines = [
                {"result": {"layoutParsingResults": [
                    {"markdown": {"text": "page one", "images": {"imgs/0.png": _IMG_URL}},
                     "outputImages": {"layout_det_res": _OUT_URL}}
                ]}},
                {"result": {"layoutParsingResults": [
                    {"markdown": {"text": "page two", "images": {}}, "outputImages": {}}
                ]}},
            ]
            return _Resp(text="\n".join(json.dumps(x) for x in lines))
        # image downloads
        return _Resp(content=_PNG)


def _client():
    c = PaddleOCRClient(api_url=_API, token="t", poll_interval=0.0, poll_timeout=10.0)
    c._session = _FakeSession()
    return c


def test_jobs_flow_materializes_flattened_pages(tmp_path):
    client = _client()
    resp = client.parse_pdf_bytes(b"%PDF-fake", tmp_path)

    # Two JSONL lines, one layout result each → two flattened pages.
    assert resp.markdown_texts == ["page one", "page two"]
    assert [p.read_text() for p in resp.markdown_paths] == ["page one", "page two"]

    # response.json carries the flattened single-result shape the page-asset
    # builder reads back.
    persisted = json.loads((tmp_path / "response.json").read_text())
    assert len(persisted["layoutParsingResults"]) == 2

    # Markdown image + output image both downloaded from their presigned urls.
    saved = {p.name for p in resp.image_paths}
    assert "0.png" in saved
    assert "layout_det_res_0.jpg" in saved


def test_submit_uses_bearer_auth_and_multipart(tmp_path):
    client = _client()
    client.parse_image_bytes(b"\x89PNG", tmp_path)

    posted = client._session.posted
    assert posted["url"] == _API
    assert posted["headers"]["Authorization"] == "bearer t"
    assert posted["data"]["model"] == "PaddleOCR-VL-1.6"
    assert json.loads(posted["data"]["optionalPayload"]) == {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,
    }
    # image submission → .png upload filename
    assert posted["files"]["file"][0] == "input.png"


def test_failed_job_raises(tmp_path):
    client = _client()

    def _get(url, headers=None, timeout=None, **kw):
        if url == f"{_API}/job-123":
            return _Resp(json_body={"data": {"state": "failed", "errorMsg": "boom"}})
        return _Resp(content=_PNG)

    client._session.get = _get
    try:
        client.parse_pdf_bytes(b"%PDF", tmp_path)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "boom" in str(e)
