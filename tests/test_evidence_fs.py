"""EvidenceFS emitter — invariants on a synthetic combined.md.

Machine-independent: drives the PURE :func:`build_evidence_fs` with a tiny
in-memory combined.md (two pages via the page marker) plus hand-built
passage→surface / sentence→surface dicts. No LinearRAG, no embedding model,
no NER. Asserts the structural contracts an agent's shell programs rely on:

  * passage / sentence char spans slice back to the exact text,
  * surface_passage edges reference only declared surfaces + passages,
  * the relation-free co-occurrence bridge fires AND respects its degree cap,
  * every TSV field is tab/newline-free (so awk/join never mis-splits),
  * documents.tsv + a sliced page file exist.
"""
import hashlib

import pytest

from ingestion.index.linear_rag.evidence_fs import MAX_COOC_SURFACES, build_evidence_fs
from ingestion.index.linear_rag.segment import PAGE_MARKER, segment_combined_md

FILE_ID = "alpha_doc_0001"

S1 = "Einstein won the Nobel Prize."
S2 = "Curie also won the Nobel Prize."
S3 = "The award is prestigious."
# an over-cap sentence: > MAX_COOC_SURFACES distinct surfaces in one sentence
BIG_SURFS = [f"thing{i}" for i in range(MAX_COOC_SURFACES + 1)]
S_BIG = " ".join(BIG_SURFS) + "."

MD = (
    "# Alpha Policy\n"
    "\n"
    f"{S1}\n{S2}\n"          # page 1: a two-sentence passage
    "\n"
    f"{PAGE_MARKER}\n"        # page boundary
    f"{S3}\n"                # page 2: one passage
    "\n"
    f"{S_BIG}\n"             # page 2: the over-cap passage
)

P1 = f"{S1}\n{S2}"
P2 = S3
P3 = S_BIG


def _h(text: str) -> str:
    return "passage-" + hashlib.md5(text.encode()).hexdigest()


@pytest.fixture
def fs(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / FILE_ID).mkdir(parents=True)
    (corpus / FILE_ID / "combined.md").write_text(MD, encoding="utf-8")

    passage_to_entities = {
        _h(P1): ["einstein", "curie", "nobel prize"],
        _h(P2): [],
        _h(P3): BIG_SURFS,
    }
    sentence_to_entities = {
        S1: ["einstein", "nobel prize"],
        S2: ["curie", "nobel prize"],
        S3: [],
        S_BIG: BIG_SURFS,
    }
    # norm_surface -> GLiNER label; "curie" is intentionally unlabeled so
    # the emit must fall back to "" for it.
    surface_to_label = {"einstein": "person", "nobel prize": "creative work"}

    out = tmp_path / "evidence_fs"
    manifest = build_evidence_fs(
        file_ids=[FILE_ID],
        read_md=lambda fid: (corpus / fid / "combined.md").read_text("utf-8"),
        passage_to_entities=passage_to_entities,
        sentence_to_entities=sentence_to_entities,
        hash_for=lambda t: "passage-" + hashlib.md5(t.encode()).hexdigest(),
        out_dir=out,
        corpus_root=corpus,
        surface_to_label=surface_to_label,
    )
    return out, manifest


def _read_tsv(path):
    lines = path.read_text("utf-8").splitlines()
    header = lines[0].split("\t")
    return header, [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def test_files_and_documents(fs):
    out, manifest = fs
    for rel in ["README.md", "EXAMPLES.md", "manifest.json",
                "nodes/documents.tsv", "nodes/passages.tsv",
                "nodes/sentences.tsv", "nodes/surfaces.tsv",
                "nodes/mentions.tsv", "edges/surface_passage.tsv",
                "edges/surface_surface_sentence.tsv",
                "views/surface_index.tsv"]:
        assert (out / rel).exists(), rel
    assert manifest["counts"]["documents"] == 1
    # documents.tsv carries a row; combined.md is a real (greppable) copy, not a
    # symlink (rg/grep -r skip symlinks); a sliced page exists.
    _, docs = _read_tsv(out / "nodes" / "documents.tsv")
    assert len(docs) == 1 and docs[0]["title"] == FILE_ID
    combined = out / "documents" / "d_0001" / "combined.md"
    assert combined.is_file() and not combined.is_symlink()
    assert combined.read_text("utf-8") == MD
    assert (out / "documents" / "d_0001" / "pages" / "page_0001.md").exists()
    assert (out / "documents" / "d_0001" / "pages" / "page_0002.md").exists()


def test_passage_and_sentence_char_spans_are_exact(fs):
    """md[start_char:end_char] must reproduce the passage / sentence text."""
    out, _ = fs
    md = (out / "documents" / "d_0001" / "combined.md").read_text("utf-8")
    # cross-check the emitted spans directly against the segmenter source.
    spans = segment_combined_md(md)
    for span in spans:
        assert md[span.start_char:span.end_char] == span.text
        for sent in span.sentences:
            assert md[sent.start_char:sent.end_char] == sent.text
    # and the persisted sentence node spans round-trip too.
    _, sents = _read_tsv(out / "nodes" / "sentences.tsv")
    for row in sents:
        sc, ec = int(row["start_char"]), int(row["end_char"])
        assert md[sc:ec] == row["text"]


def test_surface_passage_edges_reference_declared_nodes(fs):
    out, _ = fs
    sids = {s["surface_id"] for s in _read_tsv(out / "nodes" / "surfaces.tsv")[1]}
    pids = {p["passage_id"] for p in _read_tsv(out / "nodes" / "passages.tsv")[1]}
    _, sp = _read_tsv(out / "edges" / "surface_passage.tsv")
    assert sp, "expected surface_passage edges"
    for e in sp:
        assert e["surface_id"] in sids
        assert e["passage_id"] in pids


def test_surfaces_carry_ner_label(fs):
    """surfaces.tsv ner_label is populated from surface_to_label; unlabeled = ''."""
    out, _ = fs
    _, surfaces = _read_tsv(out / "nodes" / "surfaces.tsv")
    label_by_norm = {s["surface_norm"]: s["ner_label"] for s in surfaces}
    assert label_by_norm["einstein"] == "person"
    assert label_by_norm["nobel prize"] == "creative work"
    # "curie" was not in surface_to_label → empty label, not a crash.
    assert label_by_norm["curie"] == ""


def test_cooccurrence_bridge_and_degree_cap(fs):
    out, manifest = fs
    norm2id = {s["surface_norm"]: s["surface_id"]
               for s in _read_tsv(out / "nodes" / "surfaces.tsv")[1]}
    _, cooc = _read_tsv(out / "edges" / "surface_surface_sentence.tsv")
    pairs = {frozenset((c["surface_a"], c["surface_b"])) for c in cooc}
    # einstein & nobel prize share sentence S1 -> bridged
    assert frozenset((norm2id["einstein"], norm2id["nobel prize"])) in pairs
    # the > MAX-surface sentence is skipped: no thing0/thing1 bridge
    assert frozenset((norm2id["thing0"], norm2id["thing1"])) not in pairs
    assert manifest["reconstruction"].get("cooc_sentence_skipped", 0) >= 1


def test_tsv_fields_have_no_tab_or_newline(fs):
    """Every row must split into exactly header-many columns."""
    out, _ = fs
    for tsv in (list((out / "nodes").glob("*.tsv"))
                + list((out / "edges").glob("*.tsv"))
                + list((out / "views").glob("*.tsv"))):
        lines = tsv.read_text("utf-8").splitlines()
        ncol = len(lines[0].split("\t"))
        for ln in lines[1:]:
            assert len(ln.split("\t")) == ncol, f"{tsv.name}: ragged row {ln!r}"
