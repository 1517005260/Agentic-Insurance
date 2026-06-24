"""Single-batch HTTP client for the PaddleOCR layout-parsing jobs API.

Submits one file per call to the async jobs endpoint, polls until the job
reaches ``done``, downloads the resulting JSONL and persists the artifacts it
references (markdown text, markdown images, output images) into a
caller-supplied directory. Splitting and orchestration of multiple batches
live in :class:`PdfParser`.

The jobs API is asynchronous: ``POST <api_url>`` uploads the file and returns a
``jobId``; ``GET <api_url>/<jobId>`` reports ``pending`` / ``running`` / ``done``
/ ``failed``; on ``done`` the response carries a presigned ``jsonUrl`` whose
JSONL body holds one ``layoutParsingResults`` block per source page.
"""

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config.http import make_retry_session
from config.shared import shared_session
from config.settings import (
    PADDLE_OCR_API_URL,
    PADDLE_OCR_FILE_TYPE_IMAGE,
    PADDLE_OCR_FILE_TYPE_PDF,
    PADDLE_OCR_MODEL,
    PADDLE_OCR_POLL_INTERVAL,
    PADDLE_OCR_POLL_TIMEOUT,
    PADDLE_OCR_TOKEN,
)

logger = logging.getLogger(__name__)


@dataclass
class PaddleOCRResponse:
    """Materialized response for a single batch submission.

    `markdown_paths` and `markdown_texts` are aligned to the API's
    `layoutParsingResults` order.
    """

    batch_dir: Path
    markdown_paths: List[Path] = field(default_factory=list)
    markdown_texts: List[str] = field(default_factory=list)
    image_paths: List[Path] = field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None


class PaddleOCRClient:
    """One-call wrapper around the layout-parsing endpoint."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        token: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 600.0,
        poll_interval: float = PADDLE_OCR_POLL_INTERVAL,
        poll_timeout: float = PADDLE_OCR_POLL_TIMEOUT,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_chart_recognition: bool = False,
    ):
        self.api_url = api_url or PADDLE_OCR_API_URL
        self.token = token or PADDLE_OCR_TOKEN
        self.model = model or PADDLE_OCR_MODEL
        if not self.api_url:
            raise ValueError("PaddleOCR API_URL not set (env or constructor argument).")
        if not self.token:
            raise ValueError("PaddleOCR TOKEN not set (env or constructor argument).")

        self.timeout = timeout
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_chart_recognition = use_chart_recognition
        # Process-wide pool — paddle OCR's relay is the same host across
        # ingest jobs, urllib3's connection reuse saves TLS handshake.
        self._session = shared_session(
            "paddle-ocr-default", lambda: make_retry_session()
        )

    # ------------------------------------------------------------------ public

    def parse_pdf_bytes(self, pdf_bytes: bytes, output_dir: Union[str, Path]) -> PaddleOCRResponse:
        return self._submit_and_save(pdf_bytes, PADDLE_OCR_FILE_TYPE_PDF, output_dir)

    def parse_image_bytes(
        self, image_bytes: bytes, output_dir: Union[str, Path]
    ) -> PaddleOCRResponse:
        return self._submit_and_save(image_bytes, PADDLE_OCR_FILE_TYPE_IMAGE, output_dir)

    def parse_path(self, path: Union[str, Path], output_dir: Union[str, Path]) -> PaddleOCRResponse:
        path = Path(path)
        suffix = path.suffix.lower()
        file_type = PADDLE_OCR_FILE_TYPE_PDF if suffix == ".pdf" else PADDLE_OCR_FILE_TYPE_IMAGE
        return self._submit_and_save(path.read_bytes(), file_type, output_dir)

    # ------------------------------------------------------------------ core

    def _submit_and_save(
        self, file_bytes: bytes, file_type: int, output_dir: Union[str, Path]
    ) -> PaddleOCRResponse:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        job_id = self._submit_job(file_bytes, file_type)
        jsonl_url = self._poll_until_done(job_id)
        layout_results = self._fetch_layout_results(jsonl_url)

        # Flatten the JSONL pages into the single-result shape the rest of the
        # pipeline (response.json reader, page-asset builder) already expects.
        result = {"layoutParsingResults": layout_results}
        (output_dir / "response.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        materialized = PaddleOCRResponse(batch_dir=output_dir, raw_response=result)
        for i, res in enumerate(layout_results):
            self._save_one_layout_result(i, res, output_dir, materialized)

        return materialized

    # ------------------------------------------------------------------ jobs API

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"bearer {self.token}"}

    def _submit_job(self, file_bytes: bytes, file_type: int) -> str:
        """Upload a single file and return the assigned job id."""
        filename = "input.pdf" if file_type == PADDLE_OCR_FILE_TYPE_PDF else "input.png"
        data = {
            "model": self.model,
            "optionalPayload": json.dumps(
                {
                    "useDocOrientationClassify": self.use_doc_orientation_classify,
                    "useDocUnwarping": self.use_doc_unwarping,
                    "useChartRecognition": self.use_chart_recognition,
                }
            ),
        }
        # Multipart upload — requests sets the boundary Content-Type itself, so
        # the auth header must NOT carry an explicit Content-Type.
        files = {"file": (filename, file_bytes)}

        logger.info(
            "PaddleOCR submit: %d bytes, fileType=%d, model=%s",
            len(file_bytes),
            file_type,
            self.model,
        )
        response = self._session.post(
            self.api_url,
            headers=self._auth_headers(),
            data=data,
            files=files,
            timeout=self.timeout,
        )
        response.raise_for_status()
        job_id = ((response.json() or {}).get("data") or {}).get("jobId")
        if not job_id:
            raise RuntimeError(
                f"PaddleOCR submit returned no jobId; got: {response.text[:500]}"
            )
        logger.info("PaddleOCR job submitted: %s", job_id)
        return job_id

    def _poll_until_done(self, job_id: str) -> str:
        """Poll the job until it reaches ``done`` and return its JSONL url."""
        status_url = f"{self.api_url}/{job_id}"
        deadline = time.monotonic() + self.poll_timeout
        while True:
            response = self._session.get(
                status_url, headers=self._auth_headers(), timeout=self.timeout
            )
            response.raise_for_status()
            data = (response.json() or {}).get("data") or {}
            state = data.get("state")

            if state == "done":
                json_url = (data.get("resultUrl") or {}).get("jsonUrl")
                if not json_url:
                    raise RuntimeError(
                        f"PaddleOCR job {job_id} done but carried no jsonUrl."
                    )
                return json_url
            if state == "failed":
                raise RuntimeError(
                    f"PaddleOCR job {job_id} failed: {data.get('errorMsg')}"
                )

            progress = data.get("extractProgress") or {}
            logger.info(
                "PaddleOCR job %s state=%s pages=%s/%s",
                job_id,
                state,
                progress.get("extractedPages"),
                progress.get("totalPages"),
            )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"PaddleOCR job {job_id} not done after {self.poll_timeout}s "
                    f"(last state={state})."
                )
            time.sleep(self.poll_interval)

    def _fetch_layout_results(self, jsonl_url: str) -> List[Dict[str, Any]]:
        """Download the result JSONL and flatten its layout-parsing blocks.

        The url is presigned, so it is fetched without the auth header. Each
        non-empty line is ``{"result": {"layoutParsingResults": [...]}}``; the
        per-line blocks are concatenated in document order.
        """
        response = self._session.get(jsonl_url, timeout=self.timeout)
        response.raise_for_status()

        results: List[Dict[str, Any]] = []
        for line in response.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            result = json.loads(line).get("result") or {}
            results.extend(result.get("layoutParsingResults") or [])
        return results

    def _save_one_layout_result(
        self, i: int, res: Dict[str, Any], output_dir: Path, materialized: PaddleOCRResponse
    ) -> None:
        markdown_block = res.get("markdown") or {}
        text = markdown_block.get("text", "") or ""

        md_path = output_dir / f"doc_{i}.md"
        md_path.write_text(text, encoding="utf-8")
        materialized.markdown_paths.append(md_path)
        materialized.markdown_texts.append(text)

        for rel_path, payload in (markdown_block.get("images") or {}).items():
            saved = self._save_image_payload(payload, output_dir / rel_path)
            if saved is not None:
                materialized.image_paths.append(saved)

        for name, payload in (res.get("outputImages") or {}).items():
            target = output_dir / "output_images" / f"{name}_{i}.jpg"
            saved = self._save_image_payload(payload, target)
            if saved is not None:
                materialized.image_paths.append(saved)

    def _save_image_payload(self, payload: Any, dest: Path) -> Optional[Path]:
        """Materialize a markdown.images / outputImages value to disk.

        The API may deliver either a remote URL or an inline base64 string;
        both shapes are handled here.
        """
        if not isinstance(payload, str) or not payload:
            return None

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if payload.startswith("http://") or payload.startswith("https://"):
                # Short timeout: the image CDN occasionally hangs and we
                # don't want the OCR step to spend minutes per image. The
                # Markdown text is what the rest of the pipeline depends on;
                # saved images are nice-to-have.
                resp = self._session.get(payload, timeout=15)
                if resp.status_code != 200:
                    logger.warning(
                        "Failed to download image %s (status %d)", payload, resp.status_code
                    )
                    return None
                dest.write_bytes(resp.content)
            else:
                b64 = payload.split(",", 1)[1] if payload.startswith("data:") else payload
                dest.write_bytes(base64.b64decode(b64))
        except Exception as e:
            logger.warning("Failed to save image to %s: %s", dest, e)
            return None
        return dest
