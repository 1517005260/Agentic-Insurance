"""view_page resolution + the agent-loop image-injection helper.

Pins the multimodal read path that lets a vision model see a page image:
``view_page`` resolves the image next to a page's Markdown and hands its path
back; ``_image_user_message`` turns that into the ``role=user`` image block the
OpenAI-compatible API requires (it rejects images in tool messages).
"""

from pathlib import Path
from types import SimpleNamespace

from agentic.agent.base import _image_user_message
from agentic.tools.acquisition.view_page import ViewPageTool

# Smallest valid PNG (1×1, transparent) — enough to base64-encode and inspect.
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000154a24f5f0000000049454e44ae42"
    "6082"
)


def _corpus(tmp_path: Path) -> Path:
    pages = tmp_path / "documents" / "d_0001" / "pages"
    pages.mkdir(parents=True)
    (pages / "page_0001.md").write_text("page one text", encoding="utf-8")
    (pages / "page_0001.jpg").write_bytes(_PNG_1x1)
    (pages / "page_0002.md").write_text("page two, no image", encoding="utf-8")
    return tmp_path


_CTX = SimpleNamespace(add_retrieval_log=lambda **_: None)


def test_view_page_resolves_image_from_md_path(tmp_path):
    tool = ViewPageTool(corpus_root=_corpus(tmp_path))
    out, log = tool.execute(_CTX, path="documents/d_0001/pages/page_0001.md")
    assert log.get("error") is None
    assert log["image_paths"] and log["image_paths"][0].endswith("page_0001.jpg")
    assert log["image_labels"] == ["documents/d_0001/pages/page_0001.jpg"]


def test_view_page_accepts_image_path_directly(tmp_path):
    tool = ViewPageTool(corpus_root=_corpus(tmp_path))
    _, log = tool.execute(_CTX, path="documents/d_0001/pages/page_0001.jpg")
    assert log["image_paths"][0].endswith("page_0001.jpg")


def test_view_page_tolerates_omitted_documents_prefix(tmp_path):
    # root = FS root; model refers to the page without the `documents/` prefix.
    tool = ViewPageTool(corpus_root=_corpus(tmp_path))
    _, log = tool.execute(_CTX, path="d_0001/pages/page_0001.md")
    assert log["image_paths"][0].endswith("page_0001.jpg")


def test_view_page_tolerates_extra_documents_prefix(tmp_path):
    # root = documents/ (base-agent style); model adds a `documents/` prefix.
    tool = ViewPageTool(corpus_root=_corpus(tmp_path) / "documents")
    _, log = tool.execute(_CTX, path="documents/d_0001/pages/page_0001.jpg")
    assert log["image_paths"][0].endswith("page_0001.jpg")


def test_view_page_missing_image_is_not_found(tmp_path):
    tool = ViewPageTool(corpus_root=_corpus(tmp_path))
    _, log = tool.execute(_CTX, path="documents/d_0001/pages/page_0002.md")
    assert log["error"] == "not_found"


def test_view_page_rejects_path_escape(tmp_path):
    tool = ViewPageTool(corpus_root=_corpus(tmp_path))
    _, log = tool.execute(_CTX, path="../../../../etc/hosts")
    assert log["error"] == "not_found"


def test_image_user_message_builds_vision_block(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(_PNG_1x1)
    msg = _image_user_message([str(img)], ["documents/d/pages/page_0001.jpg"])
    assert msg["role"] == "user"
    text, image = msg["content"]
    assert text["type"] == "text"
    assert image["type"] == "image_url"
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_image_user_message_skips_unreadable_path():
    assert _image_user_message(["/no/such/image.jpg"], ["x"]) is None
