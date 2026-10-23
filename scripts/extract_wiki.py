"""Stream-extract prose text from a MediaWiki XML .bz2 dump (no external deps).

Usage: python scripts/extract_wiki.py <dump.xml.bz2> <out.txt>
One article per paragraph-per-line; blank line between articles.
"""
import bz2
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET


def strip_wikitext(text: str) -> str:
    # refs and their content
    text = re.sub(r"<ref[^>/]*?/>", "", text)
    text = re.sub(r"<ref[^>]*?>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*?>", "", text)
    # tables, gallery, math, pre, code blocks
    for tag in ("table", "gallery", "math", "pre", "score", "timeline"):
        text = re.sub(rf"<{tag}[^>]*?>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # templates (handle nested one level via repeated passes)
    for _ in range(3):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    # links: [[target|label]] -> label ; [[File:...]] dropped
    text = re.sub(r"\[\[(?:File|Image|Category|ചിത്രം|വർഗ്ഗം|ഫയൽ):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    # external links [url label] -> label
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    # headings and formatting markers
    text = re.sub(r"^\s*=+.*?=+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"'''?", "", text)
    # html comments and remaining tags
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    # wiki list/indent markers at line start
    text = re.sub(r"^\s*[:*#;]+\s*", "", text, flags=re.MULTILINE)
    return text


def paras(text: str):
    for block in text.split("\n\n"):
        lines = [ln.strip() for ln in block.strip().splitlines()]
        p = " ".join(ln for ln in lines if ln)
        if p:
            yield p


def main(dump_path, out_path):
    n_articles = n_paras = n_chars = 0
    with bz2.open(dump_path, "rb") as fin, open(out_path, "w", encoding="utf-8") as fout:
        ctx = ET.iterparse(fin, events=("end",))
        for _, elem in ctx:
            if elem.tag.endswith("}page") or elem.tag == "page":
                ns = elem.findtext("./{*}ns") or "0"
                if ns not in (None, "0"):
                    elem.clear()
                    continue
                title = elem.findtext("./{*}title") or ""
                if ":" in title:
                    elem.clear()
                    continue
                raw = elem.findtext(".//{*}text") or ""
                if not raw or len(raw) < 200:
                    elem.clear()
                    continue
                text = unicodedata.normalize("NFC", strip_wikitext(raw))
                got = False
                for p in paras(text):
                    # keep prose-like paragraphs only
                    if len(p) >= 80 and sum(1 for c in p if "\u0d00" <= c <= "\u0d7f") / max(len(p), 1) > 0.35:
                        fout.write(p + "\n\n")
                        n_paras += 1
                        n_chars += len(p)
                        got = True
                if got:
                    fout.write("\x1e\n\n")  # doc separator for split phase
                    n_articles += 1
                elem.clear()
    print(f"articles={n_articles} paragraphs={n_paras} chars={n_chars}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
