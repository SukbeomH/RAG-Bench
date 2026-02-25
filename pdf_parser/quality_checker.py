"""
Markdown 변환 품질 검사 유틸리티
"""

from pathlib import Path


def check_quality(md_file: str | Path) -> dict:
    """
    변환된 Markdown 파일의 품질 지표 반환.

    Args:
        md_file: Markdown 파일 경로

    Returns:
        품질 지표 딕셔너리
    """
    content = Path(md_file).read_text(encoding="utf-8")
    lines = content.splitlines()
    words = content.split()

    return {
        "word_count": len(words),
        "line_count": len(lines),
        "has_headers": "#" in content,
        "has_tables": "|" in content,
        "has_formulas": "$" in content,
        "has_code_blocks": "```" in content,
        "avg_line_length": len(content) / max(len(lines), 1),
        "empty_ratio": content.count("\n\n") / max(content.count("\n"), 1),
    }


def check_folder(output_folder: str, min_word_count: int = 50) -> list[dict]:
    """
    폴더 내 모든 Markdown 파일의 품질을 일괄 검사.

    Args:
        output_folder: Markdown 파일이 있는 폴더 경로
        min_word_count: 최소 단어 수 (미달 시 경고)

    Returns:
        파일별 품질 지표 목록
    """
    results = []
    md_files = list(Path(output_folder).glob("*.md"))

    if not md_files:
        print(f"⚠ Markdown 파일 없음: {output_folder}")
        return results

    for md_file in md_files:
        metrics = check_quality(md_file)
        metrics["file"] = md_file.name
        results.append(metrics)

        status = "✓" if metrics["word_count"] >= min_word_count else "⚠"
        print(
            f"{status} {md_file.name}: "
            f"단어 {metrics['word_count']}개 | "
            f"헤더 {'있음' if metrics['has_headers'] else '없음'} | "
            f"표 {'있음' if metrics['has_tables'] else '없음'} | "
            f"수식 {'있음' if metrics['has_formulas'] else '없음'}"
        )

    return results


if __name__ == "__main__":
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "./md_output"
    check_folder(folder)
