from __future__ import annotations

import copy
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from deep_translator import GoogleTranslator
from deep_translator.exceptions import TranslationNotFound

XLIFF_NS = "urn:oasis:names:tc:xliff:document:1.2"
NSMAP = {"x": XLIFF_NS}
ET.register_namespace("", XLIFF_NS)


@dataclass
class TranslationOptions:
    target_lang: str                 # e.g. "fr", "de"
    source_lang: Optional[str] = None # e.g. "en" or None/"auto"
    set_file_target_language: Optional[str] = None
    prefix_target: str = ""
    suffix_target: str = ""
    glossary: Optional[Dict[str, str]] = None

    # POC reliability knobs
    retries: int = 2
    retry_backoff_s: float = 0.35
    fallback_to_source_on_error: bool = True
    mark_fallback_in_target: bool = False  # if True, wraps failed nodes with [[...]]


@dataclass
class SegmentResult:
    unit_id: str
    source_text: str
    target_text: str
    flagged: bool
    flag_reasons: List[str]


def extract_visible_text(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    text = "".join(elem.itertext())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalise_lang(code: Optional[str]) -> str:
    return (code or "").strip().lower() or "auto"


def _apply_glossary(text: str, glossary: Optional[Dict[str, str]]) -> str:
    if not glossary:
        return text
    out = text
    for k in sorted(glossary.keys(), key=len, reverse=True):
        out = out.replace(k, glossary[k])
    return out


def _looks_untranslated(source: str, target: str) -> bool:
    s = re.sub(r"\s+", " ", (source or "").lower().strip())
    t = re.sub(r"\s+", " ", (target or "").lower().strip())
    return bool(s and t and s == t)


def _split_ws(s: str) -> Tuple[str, str, str]:
    if s is None:
        return "", "", ""
    m = re.match(r"^(\s*)(.*?)(\s*)$", s, flags=re.DOTALL)
    if not m:
        return "", s, ""
    return m.group(1), m.group(2), m.group(3)


def _collect_text_nodes(root: ET.Element) -> List[Tuple[ET.Element, str, str, str, str]]:
    """
    Collect translatable text nodes within an element tree:
      - (element, "text"/"tail", leading_ws, core, trailing_ws)
    """
    nodes: List[Tuple[ET.Element, str, str, str, str]] = []

    if root.text is not None and root.text != "":
        lead, core, trail = _split_ws(root.text)
        nodes.append((root, "text", lead, core, trail))

    for child in list(root):
        if child.text is not None and child.text != "":
            lead, core, trail = _split_ws(child.text)
            nodes.append((child, "text", lead, core, trail))
        if child.tail is not None and child.tail != "":
            lead, core, trail = _split_ws(child.tail)
            nodes.append((child, "tail", lead, core, trail))

        nodes.extend(_collect_text_nodes(child))

    return nodes


def _set_node_text(node: ET.Element, which: str, value: str) -> None:
    if which == "text":
        node.text = value
    elif which == "tail":
        node.tail = value
    else:
        raise ValueError(f"Unknown node part: {which}")


def _safe_translate_one(
    translator: GoogleTranslator,
    text: str,
    opts: TranslationOptions,
) -> Tuple[str, bool]:
    """
    Returns (translated_text, used_fallback)
    """
    if not text.strip():
        return "", False

    last_err: Optional[Exception] = None
    for attempt in range(opts.retries + 1):
        try:
            return translator.translate(text), False
        except TranslationNotFound as e:
            last_err = e
        except Exception as e:  # network / parsing / transient
            last_err = e

        # backoff before retry (except after final attempt)
        if attempt < opts.retries:
            time.sleep(opts.retry_backoff_s * (attempt + 1))

    # fallback
    if opts.fallback_to_source_on_error:
        if opts.mark_fallback_in_target:
            return f"[[{text}]]", True
        return text, True

    # if no fallback allowed, re-raise the last error
    assert last_err is not None
    raise last_err


def translate_texts_deep_safe(
    texts: List[str],
    target_lang: str,
    source_lang: Optional[str],
    opts: TranslationOptions,
) -> Tuple[List[str], int]:
    """
    Translate list of strings safely. Returns (translations, failed_count)
    """
    translator = GoogleTranslator(
        source=_normalise_lang(source_lang),
        target=_normalise_lang(target_lang),
    )

    out: List[str] = []
    failed = 0
    for t in texts:
        translated, used_fallback = _safe_translate_one(translator, t, opts)
        if used_fallback:
            failed += 1
        out.append(translated)
    return out, failed


def _replace_target_with_preserved_markup(
    tu: ET.Element,
    source_elem: ET.Element,
    translated_texts: List[str],
    opts: TranslationOptions,
) -> None:
    # Copy markup
    target_subtree = copy.deepcopy(source_elem)
    target_subtree.tag = f"{{{XLIFF_NS}}}target"

    nodes = _collect_text_nodes(target_subtree)

    for (node, which, lead, _core, trail), translated in zip(nodes, translated_texts):
        translated = _apply_glossary(translated, opts.glossary)
        translated = f"{opts.prefix_target}{translated}{opts.suffix_target}"
        _set_node_text(node, which, f"{lead}{translated}{trail}")

    existing_target = tu.find("x:target", NSMAP)
    if existing_target is not None:
        tu.remove(existing_target)

    children = list(tu)
    src_index = 0
    for i, c in enumerate(children):
        if c is source_elem:
            src_index = i
            break
    tu.insert(src_index + 1, target_subtree)


def translate_xliff_bytes_google(
    xliff_bytes: bytes,
    opts: TranslationOptions,
    batch_size: int = 30,
    progress_callback=None,
) -> Tuple[bytes, List[SegmentResult]]:
    """
    Preserves formatting by copying <source> markup into <target> and translating only text nodes.
    Fault tolerant: translation failures fall back instead of crashing the run.
    """
    root = ET.fromstring(xliff_bytes)
    trans_units = root.findall(".//x:trans-unit", NSMAP)

    tus_for_progress = []
    for i, tu in enumerate(trans_units):
        src = tu.find("x:source", NSMAP)
        if src is None:
            continue
        tus_for_progress.append((tu, src, tu.get("id") or f"row-{i+1}"))

    total_units = len(tus_for_progress)
    results: List[SegmentResult] = []

    for start in range(0, total_units, batch_size):
        end = min(start + batch_size, total_units)
        batch_tus = tus_for_progress[start:end]

        batch_node_maps = []
        batch_texts = []

        for tu, src_elem, unit_id in batch_tus:
            src_nodes = _collect_text_nodes(src_elem)
            cores = [core for (_node, _which, _lead, core, _trail) in src_nodes]
            batch_node_maps.append((tu, src_elem, unit_id, src_nodes))
            batch_texts.extend(cores)

        translated_flat, failed_count = translate_texts_deep_safe(
            texts=batch_texts,
            target_lang=opts.target_lang,
            source_lang=opts.source_lang,
            opts=opts,
        )

        idx = 0
        for tu, src_elem, unit_id, src_nodes in batch_node_maps:
            n = len(src_nodes)
            translated_slice = translated_flat[idx: idx + n]
            idx += n

            _replace_target_with_preserved_markup(
                tu=tu,
                source_elem=src_elem,
                translated_texts=translated_slice,
                opts=opts,
            )

            src_text = extract_visible_text(src_elem)
            tgt_elem = tu.find("x:target", NSMAP)
            tgt_text = extract_visible_text(tgt_elem)

            reasons: List[str] = []
            if _looks_untranslated(src_text, tgt_text):
                reasons.append("Looks untranslated")

            # If any node in this TU used fallback, we can't easily know per TU from the flat list
            # without extra bookkeeping; but we can still tell the user globally via UI if needed.

            results.append(
                SegmentResult(
                    unit_id=unit_id,
                    source_text=src_text,
                    target_text=tgt_text,
                    flagged=bool(reasons),
                    flag_reasons=reasons,
                )
            )

        if progress_callback:
            progress_callback(end, total_units)

    if opts.set_file_target_language:
        file_elem = root.find("x:file", NSMAP)
        if file_elem is not None:
            file_elem.set("target-language", opts.set_file_target_language)

    out_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return out_bytes, results
