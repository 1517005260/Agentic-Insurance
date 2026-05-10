"""Single-batch HTTP client for the PP-StructureV3 layout-parsing API.

Submits one base64-encoded file per call and persists the artifacts referenced
in the response (markdown text, markdown images, output images) into a
caller-supplied directory. Splitting and orchestration of multiple batches
live in :class:`PdfParser`.
"""

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config.http import make_retry_session
from config.shared import shared_session
from config.settings import (
    PADDLE_OCR_API_URL,
    PADDLE_OCR_FILE_TYPE_IMAGE,
    PADDLE_OCR_FILE_TYPE_PDF,
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
        timeout: float = 600.0,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_chart_recognition: bool = False,
    ):
        self.api_url = api_url or PADDLE_OCR_API_URL
        self.token = token or PADDLE_OCR_TOKEN
        if not self.api_url:
            raise ValueError("PaddleOCR API_URL not set (env or constructor argument).")
        if not self.token:
            raise ValueError("PaddleOCR TOKEN not set (env or constructor argument).")

        self.timeout = timeout
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

        payload = {
            "file": base64.b64encode(file_bytes).decode("ascii"),
            "fileType": file_type,
            "useDocOrientationClassify": self.use_doc_orientation_classify,
            "useDocUnwarping": self.use_doc_unwarping,
            "useChartRecognition": self.use_chart_recognition,
        }
        headers = {
            "Authorization": f"token {self.token}",
            "Content-Type": "application/json",
        }

        logger.info(
            "PaddleOCR submit: %d bytes, fileType=%d, output_dir=%s",
            len(file_bytes),
            file_type,
            output_dir,
        )
        response = self._session.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        result = response.json().get("result")
        if result is None:
            raise RuntimeError(
                f"PaddleOCR response missing 'result' field; got: {response.text[:500]}"
            )

        # Preserve the full JSON for replay / debugging.
        (output_dir / "response.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        materialized = PaddleOCRResponse(batch_dir=output_dir, raw_response=result)
        for i, res in enumerate(result.get("layoutParsingResults", []) or []):
            self._save_one_layout_result(i, res, output_dir, materialized)

        return materialized

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
