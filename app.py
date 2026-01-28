import time
import re
from typing import Dict, Optional, List

import pandas as pd
import streamlit as st

from xliff_translate_google import (
    TranslationOptions,
    translate_xliff_bytes_google,
)

st.set_page_config(
    page_title="WD-40 Digital • Rise XLF Translator",
    page_icon="🛠️",
    layout="wide",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
      .hero {
        border-radius: 20px;
        padding: 18px;
        border: 1px solid rgba(0,0,0,0.08);
        background: linear-gradient(90deg, rgba(0,56,168,0.14), rgba(255,193,7,0.18));
      }
      .hero-title { font-size: 1.6rem; font-weight: 800; }
      .hero-sub { margin-top: 6px; opacity: 0.85; }
      .card {
        border-radius: 18px;
        padding: 16px;
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(255,255,255,0.6);
      }
      .stDataFrame { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-title">WD-40 Digital • Rise XLF Translator</div>
      <div class="hero-sub">
        POC translator for Rise-exported XLIFF 1.2. QA is tuned to show proof of translation, not just boilerplate IDs.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------- Helpers ----------
LANG_PRESETS = [
    ("French (FR)", "fr", "fr-fr"),
    ("German (DE)", "de", "de-de"),
    ("Spanish (ES)", "es", "es-es"),
    ("Italian (IT)", "it", "it-it"),
]


def parse_glossary(raw: str) -> Optional[Dict[str, str]]:
    mapping: Dict[str, str] = {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=>" not in line:
            continue
        left, right = line.split("=>", 1)
        left = left.strip()
        right = right.strip()
        if left:
            mapping[left] = right
    return mapping or None


def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def looks_like_id_or_token(s: str) -> bool:
    s2 = (s or "").strip()
    if not s2:
        return True
    if "items|id:" in s2 or "answers|id:" in s2:
        return True
    if s2.startswith("cm") and len(s2) >= 12 and re.fullmatch(r"[a-zA-Z0-9]+", s2):
        return True
    if re.fullmatch(r"[a-f0-9]{12,}", s2.lower()):
        return True
    return False


def looks_like_numberish(s: str) -> bool:
    s2 = (s or "").strip()
    if not s2:
        return True
    if re.fullmatch(r"\d+(\.\d+)?%?", s2):
        return True
    if re.fullmatch(r"\$?\d+(\.\d+)?[KMB]?", s2, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"(up to\s*)?\$?\d+(\.\d+)?[KMB]?", s2, flags=re.IGNORECASE):
        return True
    return False


def qa_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Source"] = out["Source"].fillna("")
    out["Target"] = out["Target"].fillna("")

    out["Empty source"] = out["Source"].str.strip().eq("")
    out["Empty target"] = out["Target"].str.strip().eq("")

    out["Changed"] = out.apply(lambda r: norm_text(r["Source"]) != norm_text(r["Target"]), axis=1)
    out["Non-linguistic"] = out["Source"].apply(
        lambda s: looks_like_id_or_token(s) or looks_like_numberish(s) or len((s or "").strip()) <= 3
    )

    src_len = out["Source"].str.len().clip(lower=1)
    tgt_len = out["Target"].str.len()
    ratio = (tgt_len / src_len).astype(float)
    out["Length ratio"] = ratio
    out["Ratio flag"] = (src_len >= 12) & (tgt_len >= 12) & ((ratio < 0.45) | (ratio > 2.2))

    out["Flagged_backend"] = out.get("Flagged_backend", False)
    out["Looks untranslated (linguistic)"] = out["Flagged_backend"] & (~out["Non-linguistic"])

    out["Flagged"] = out["Looks untranslated (linguistic)"] | out["Empty target"] | out["Ratio flag"]

    def build_reasons(row):
        reasons: List[str] = []
        if row.get("Looks untranslated (linguistic)"):
            reasons.append("Looks untranslated")
        if row.get("Empty source"):
            reasons.append("Empty source")
        if row.get("Empty target"):
            reasons.append("Empty target")
        if row.get("Ratio flag"):
            reasons.append(f"Length ratio {row['Length ratio']:.2f}×")
        return "; ".join(reasons)

    out["Reasons"] = out.apply(build_reasons, axis=1)
    return out


def safe_key(base: str, extra: Optional[str]) -> str:
    """
    Makes a stable-ish unique key based on a base string + optional extra identifier.
    """
    extra = (extra or "").strip()
    extra = re.sub(r"[^a-zA-Z0-9_-]+", "_", extra)[:80]
    return f"{base}__{extra}" if extra else base


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Configuration")

    preset_label = st.selectbox("Target language", [p[0] for p in LANG_PRESETS], index=0)
    preset = next(p for p in LANG_PRESETS if p[0] == preset_label)
    target_lang, locale = preset[1], preset[2]

    source_lang = st.text_input("Source language", value="en").strip() or None
    set_file_target_language = st.text_input("XLIFF target-language", value=locale).strip() or None

    st.divider()
    batch_size = st.slider("Batch size", 10, 80, 40, 5)
    polite_delay = st.slider("Polite delay per batch (seconds)", 0.0, 1.5, 0.0, 0.1)

    with st.expander("Brand / QA tools", expanded=False):
        prefix_target = st.text_input("Prefix every target", "")
        suffix_target = st.text_input("Suffix every target", "")
        glossary_raw = st.text_area("Glossary (from => to)", height=120)

# ---------- Main ----------
uploaded = st.file_uploader("Upload a Rise-exported `.xlf` / `.xliff`", type=["xlf", "xliff"])

if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

raw = uploaded.read()

c1, c2, c3, c4 = st.columns(4)
c1.metric("File", uploaded.name)
c2.metric("Size", f"{len(raw):,} bytes")
c3.metric("Source", source_lang or "auto")
c4.metric("Target", target_lang)

opts = TranslationOptions(
    target_lang=target_lang,
    source_lang=source_lang,
    set_file_target_language=set_file_target_language,
    prefix_target=prefix_target,
    suffix_target=suffix_target,
    glossary=parse_glossary(glossary_raw),
)

if "df" not in st.session_state:
    st.session_state.df = None
if "out_bytes" not in st.session_state:
    st.session_state.out_bytes = None
if "out_name" not in st.session_state:
    st.session_state.out_name = None

tab_translate, tab_review = st.tabs(["🚀 Translate + QA", "🔎 Review"])

with tab_translate:
    left, right = st.columns([1.1, 0.9], vertical_alignment="top")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Input preview**")
        with st.expander("Show first ~4,000 characters"):
            st.code(raw[:4000].decode("utf-8", errors="replace"), language="xml")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Translate**")
        run = st.button("Translate XLF", type="primary", use_container_width=True, key="run_translate_btn")

        if run:
            prog = st.progress(0, text="Starting…")
            status = st.empty()

            def cb(done, total):
                pct = int((done / max(1, total)) * 100)
                prog.progress(pct, text=f"Translating… {done:,}/{total:,} segments")
                status.write(f"Processed **{done:,}** / **{total:,}** segments")
                if polite_delay:
                    time.sleep(polite_delay)

            out_bytes, results = translate_xliff_bytes_google(
                raw,
                opts=opts,
                batch_size=batch_size,
                progress_callback=cb,
            )

            out_name = f"{uploaded.name.rsplit('.',1)[0]}-{target_lang}.xlf"
            st.session_state.out_bytes = out_bytes
            st.session_state.out_name = out_name

            df = pd.DataFrame([r.__dict__ for r in results])
            df["Flagged_backend"] = df["flagged"]
            df = df.rename(columns={"unit_id": "ID", "source_text": "Source", "target_text": "Target", "flagged": "Flagged"})
            df = qa_flags(df)

            st.session_state.df = df

            prog.progress(100, text="Complete ✅")
            status.success("Done.")

        # IMPORTANT: download button can appear even if run isn't pressed (state-based),
        # so give it a unique key
        if st.session_state.out_bytes:
            st.download_button(
                "⬇️ Download translated XLF",
                data=st.session_state.out_bytes,
                file_name=st.session_state.out_name or "translated.xlf",
                mime="application/octet-stream",
                use_container_width=True,
                key=safe_key("download_xlf_translate_tab", st.session_state.out_name),
            )

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.df is not None:
        df = st.session_state.df

        st.markdown("## Proof it’s working: translated sample")
        changed_df = df[df["Changed"] == True]  # noqa: E712

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Segments", f"{len(df):,}")
        s2.metric("Changed", f"{len(changed_df):,}")
        s3.metric("Unchanged", f"{int((~df['Changed']).sum()):,}")
        s4.metric("Flagged", f"{int(df['Flagged'].sum()):,}")

        sample_n = st.slider("Sample size", 5, 50, 12, 1, key="sample_n_slider")
        if len(changed_df) > 0:
            sample = changed_df.sample(n=min(sample_n, len(changed_df)), random_state=42)
            st.dataframe(
                sample[["ID", "Source", "Target"]],
                use_container_width=True,
                hide_index=True,
                height=420,
            )
        else:
            st.warning("No changed segments detected (unexpected if output contains translated content).")

        st.markdown("## QA workbench")
        view_mode = st.radio(
            "View",
            ["Changed (recommended)", "Flagged", "All", "Unchanged"],
            horizontal=True,
            key="qa_view_mode",
        )
        search = st.text_input("Search", value="", placeholder="Search source / target / reasons…", key="qa_search")
        rows = st.selectbox("Rows", [200, 500, 1000, 2500, 5000, 10000], index=2, key="qa_rows")
        height = st.slider("Table height", 450, 1200, 900, 50, key="qa_height")

        view = df.copy()
        if view_mode.startswith("Changed"):
            view = view[view["Changed"] == True]  # noqa: E712
        elif view_mode == "Flagged":
            view = view[view["Flagged"] == True]  # noqa: E712
        elif view_mode == "Unchanged":
            view = view[view["Changed"] == False]  # noqa: E712

        if search.strip():
            q = search.strip().lower()
            view = view[
                view["Source"].fillna("").str.lower().str.contains(q)
                | view["Target"].fillna("").str.lower().str.contains(q)
                | view["Reasons"].fillna("").str.lower().str.contains(q)
            ]

        st.dataframe(
            view[["ID", "Changed", "Flagged", "Reasons", "Source", "Target"]].head(rows),
            use_container_width=True,
            hide_index=True,
            height=height,
        )

with tab_review:
    st.subheader("Review")

    if st.session_state.df is None:
        st.info("Run a translation first.")
    else:
        df = st.session_state.df
        st.dataframe(
            df[["ID", "Changed", "Flagged", "Reasons", "Source", "Target"]].head(5000),
            use_container_width=True,
            height=900,
            hide_index=True,
        )

        # Second download button → MUST have a different key
        if st.session_state.out_bytes:
            st.download_button(
                "⬇️ Download translated XLF",
                data=st.session_state.out_bytes,
                file_name=st.session_state.out_name or "translated.xlf",
                mime="application/octet-stream",
                use_container_width=True,
                key=safe_key("download_xlf_review_tab", st.session_state.out_name),
            )
