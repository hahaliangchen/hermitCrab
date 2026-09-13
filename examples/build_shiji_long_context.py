"""Build long-context MLM samples from selected Shiji chapters.

The source file is already whitespace-tokenized.  This utility keeps each
chapter separate and greedily groups source sentences into samples of about
100 word tokens, preserving the original punctuation and sentence order.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "examples" / "shiji_baihua.txt"
OUTPUT_PATH = ROOT / "examples" / "shiji_baihua_zhangchen_gaozu_long_context.txt"

TARGET_WORDS = 100


def normalized_tokens(line: str) -> list[str]:
    return line.split()


def chapter_lines(lines: list[str], start_heading: str, end_heading: str) -> list[str]:
    start = lines.index(start_heading) + 1
    end = lines.index(end_heading, start)
    return [line.strip() for line in lines[start:end] if line.strip()]


def make_chunks(source_lines: list[str]) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for line in source_lines:
        tokens = normalized_tokens(line)
        if not tokens:
            continue

        # No source sentence in the selected chapters is this long today,
        # but splitting here keeps max_length=128 safe if the source changes.
        while len(tokens) > TARGET_WORDS:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_size = 0
            chunks.append(" ".join(tokens[:TARGET_WORDS]))
            tokens = tokens[TARGET_WORDS:]

        if not tokens:
            continue
        if current and current_size + len(tokens) > TARGET_WORDS:
            chunks.append(" ".join(current))
            current = []
            current_size = 0
        current.extend(tokens)
        current_size += len(tokens)

    if current:
        chunks.append(" ".join(current))
    return chunks


def main() -> None:
    lines = SOURCE_PATH.read_text(encoding="utf-8").splitlines()
    zhangchen = chapter_lines(lines, "张耳 陈余 列传", "淮阴 侯 列传")
    gaozu = chapter_lines(lines, "高祖 本纪", "孝武 本纪")
    zhangchen_chunks = make_chunks(zhangchen)
    gaozu_chunks = make_chunks(gaozu)
    chunks = zhangchen_chunks + gaozu_chunks
    OUTPUT_PATH.write_text("\n".join(chunks) + "\n", encoding="utf-8")

    sizes = [len(chunk.split()) for chunk in chunks]
    print(f"source: {SOURCE_PATH}")
    print(f"output: {OUTPUT_PATH}")
    print(f"张耳陈余: {len(zhangchen_chunks)} samples")
    print(f"高祖本纪: {len(gaozu_chunks)} samples")
    print(
        "total: {} samples, word_tokens min={}, max={}, avg={:.2f}".format(
            len(chunks), min(sizes), max(sizes), sum(sizes) / len(sizes)
        )
    )


if __name__ == "__main__":
    main()
