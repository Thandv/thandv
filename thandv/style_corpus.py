"""Bundled public-domain style guides + literary corpora for the writer
persona's local RAG store.

Every snippet is either US public domain (pre-1929 publication or
government work) or composed in-house and covered by the repo's LICENSE.
Provenance is recorded per snippet so the source/license lives in the
file, not in someone's head.

The bundle is small on purpose. The goal is to give the writer persona a
few high-signal style references the RAG can reach during editing tasks
— not to ship a literary library.

Usage:
    thandv writer bundle-style-corpus           # ingest into writer corpus
    thandv writer bundle-style-corpus --clear   # wipe + re-ingest
"""

from __future__ import annotations

from dataclasses import dataclass

from thandv import rag


@dataclass(frozen=True)
class StyleSnippet:
    title: str
    source: str  # bibliographic note: author, year, license rationale
    license: str
    text: str


# --- Public-domain literary excerpts ------------------------------------

_GETTYSBURG = StyleSnippet(
    title="Lincoln, Gettysburg Address",
    source="Abraham Lincoln, 1863. Public domain (US federal address, pre-1929).",
    license="public-domain",
    text=(
        "Four score and seven years ago our fathers brought forth on this "
        "continent, a new nation, conceived in Liberty, and dedicated to the "
        "proposition that all men are created equal.\n\n"
        "Now we are engaged in a great civil war, testing whether that nation, "
        "or any nation so conceived and so dedicated, can long endure. We are "
        "met on a great battle-field of that war. We have come to dedicate a "
        "portion of that field, as a final resting place for those who here "
        "gave their lives that that nation might live. It is altogether fitting "
        "and proper that we should do this.\n\n"
        "But, in a larger sense, we can not dedicate -- we can not consecrate "
        "-- we can not hallow -- this ground. The brave men, living and dead, "
        "who struggled here, have consecrated it, far above our poor power to "
        "add or detract."
    ),
)


_TWAIN_HUCK = StyleSnippet(
    title="Twain, Huckleberry Finn (opening)",
    source="Mark Twain, 1884. Public domain.",
    license="public-domain",
    text=(
        "You don't know about me without you have read a book by the name of "
        "The Adventures of Tom Sawyer; but that ain't no matter. That book "
        "was made by Mr. Mark Twain, and he told the truth, mainly. There "
        "was things which he stretched, but mainly he told the truth. That "
        "is nothing. I never seen anybody but lied one time or another, "
        "without it was Aunt Polly, or the widow, or maybe Mary."
    ),
)


_SHAKESPEARE_18 = StyleSnippet(
    title="Shakespeare, Sonnet 18",
    source="William Shakespeare, c.1609. Public domain.",
    license="public-domain",
    text=(
        "Shall I compare thee to a summer's day?\n"
        "Thou art more lovely and more temperate:\n"
        "Rough winds do shake the darling buds of May,\n"
        "And summer's lease hath all too short a date:\n"
        "Sometime too hot the eye of heaven shines,\n"
        "And often is his gold complexion dimm'd;\n"
        "And every fair from fair sometime declines,\n"
        "By chance, or nature's changing course untrimm'd;\n"
        "But thy eternal summer shall not fade,\n"
        "Nor lose possession of that fair thou ow'st;\n"
        "Nor shall Death brag thou wander'st in his shade,\n"
        "When in eternal lines to time thou grow'st;\n"
        "So long as men can breathe or eyes can see,\n"
        "So long lives this, and this gives life to thee."
    ),
)


# --- Public-domain style guide excerpts (Strunk 1918) -------------------

_STRUNK_OMIT_WORDS = StyleSnippet(
    title="Strunk, Elements of Style - Omit needless words",
    source="William Strunk Jr., The Elements of Style, 1918. Public domain.",
    license="public-domain",
    text=(
        "Vigorous writing is concise. A sentence should contain no unnecessary "
        "words, a paragraph no unnecessary sentences, for the same reason "
        "that a drawing should have no unnecessary lines and a machine no "
        "unnecessary parts. This requires not that the writer make all his "
        "sentences short, or that he avoid all detail and treat his subjects "
        "only in outline, but that every word tell."
    ),
)


_STRUNK_DEFINITE = StyleSnippet(
    title="Strunk, Elements of Style - Use definite, specific, concrete language",
    source="William Strunk Jr., The Elements of Style, 1918. Public domain.",
    license="public-domain",
    text=(
        "Prefer the specific to the general, the definite to the vague, the "
        "concrete to the abstract. The greatest writers -- Homer, Dante, "
        "Shakespeare -- are effective largely because they deal in particulars "
        "and report the details that matter. Their words call up pictures."
    ),
)


_STRUNK_ACTIVE = StyleSnippet(
    title="Strunk, Elements of Style - Use the active voice",
    source="William Strunk Jr., The Elements of Style, 1918. Public domain.",
    license="public-domain",
    text=(
        "The active voice is usually more direct and vigorous than the "
        "passive. Many a tame sentence of description or exposition can be "
        "made lively and emphatic by substituting a transitive in the active "
        "voice for some such perfunctory expression as 'there is' or 'could "
        "be heard.'"
    ),
)


# --- Original house-style guide -----------------------------------------

_THANDV_HOUSE_STYLE = StyleSnippet(
    title="Thandv writer house style",
    source="Composed in-house for Thandv.",
    license="MIT (per repo LICENSE)",
    text=(
        "Write with restraint.\n\n"
        "- One idea per sentence; one sentence per idea.\n"
        "- Cut intensifiers ('very', 'really', 'quite') unless they earn their keep.\n"
        "- Prefer concrete nouns and active verbs to abstract nominalisations.\n"
        "- Never write 'I think' or 'in my opinion' -- the reader knows the source.\n"
        "- When editing, propose the specific replacement text. Don't say "
        "'consider rephrasing'; show the rephrase.\n"
        "- For long pieces: outline before drafting. Sections, then "
        "paragraphs, then sentences.\n"
    ),
)


STYLE_CORPUS: tuple[StyleSnippet, ...] = (
    _GETTYSBURG,
    _TWAIN_HUCK,
    _SHAKESPEARE_18,
    _STRUNK_OMIT_WORDS,
    _STRUNK_DEFINITE,
    _STRUNK_ACTIVE,
    _THANDV_HOUSE_STYLE,
)


WRITER_PERSONA = "writer"
SOURCE_PREFIX = "style_corpus::"


def bundle_writer_corpus(
    persona: str = WRITER_PERSONA, *, clear: bool = False
) -> int:
    """Ingest every snippet into the persona's RAG corpus.

    Returns the number of chunks written. Requires Ollama's embed model
    (same as `thandv ingest`). If `clear=True`, wipes the persona's
    corpus before ingest -- so repeat calls stay idempotent rather than
    appending duplicates.
    """
    if not rag.embed_model_available():
        raise RuntimeError(
            f"embed model {rag.EMBED_MODEL!r} not available on Ollama. "
            "Run `ollama pull nomic-embed-text` first."
        )

    if clear:
        rag.clear_corpus(persona)

    total = 0
    for snippet in STYLE_CORPUS:
        source = f"{SOURCE_PREFIX}{snippet.title}"
        total += rag.ingest_text(snippet.text, source=source, persona=persona)
    return total
