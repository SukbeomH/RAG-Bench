"""
01. PDF 파싱 및 청킹 스크립트

docs/ 디렉토리의 PDF 파일을 파싱하고, 청킹하여 corpus.parquet를 생성한다.
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "autorag_benchmark"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIG_DIR = BENCHMARK_DIR / "config"
PARSE_PROJECT_DIR = BENCHMARK_DIR / "parse_project"
CHUNK_PROJECT_DIR = BENCHMARK_DIR / "chunk_project"


def check_pdf_files():
    """docs/ 디렉토리에 PDF 파일이 있는지 확인"""
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[ERROR] {DOCS_DIR} 디렉토리에 PDF 파일이 없습니다.")
        print("PDF 파일을 docs/ 디렉토리에 넣어주세요.")
        sys.exit(1)
    print(f"[INFO] 발견된 PDF 파일 {len(pdf_files)}개:")
    for f in pdf_files:
        print(f"  - {f.name}")
    return pdf_files


def run_parsing():
    """Stage 1: PDF → 파싱된 텍스트"""
    from autorag.parser import Parser

    print("\n" + "=" * 60)
    print("Stage 1: PDF 파싱")
    print("=" * 60)

    parser = Parser(
        data_path_glob=str(DOCS_DIR / "*.pdf"),
        project_dir=str(PARSE_PROJECT_DIR),
    )
    parser.start_parsing(str(CONFIG_DIR / "parse_config.yaml"))

    # 파싱 결과 확인
    parsed_files = list(PARSE_PROJECT_DIR.glob("**/*.parquet"))
    if parsed_files:
        print(f"[SUCCESS] 파싱 완료: {parsed_files[0]}")
        return parsed_files[0]
    else:
        print("[ERROR] 파싱 결과 파일이 생성되지 않았습니다.")
        sys.exit(1)


def run_chunking(parsed_data_path: Path):
    """Stage 2: 파싱된 텍스트 → 청킹된 코퍼스"""
    from autorag.chunker import Chunker

    print("\n" + "=" * 60)
    print("Stage 2: 텍스트 청킹")
    print("=" * 60)

    chunker = Chunker.from_parquet(
        parsed_data_path=str(parsed_data_path),
        project_dir=str(CHUNK_PROJECT_DIR),
    )
    chunker.start_chunking(str(CONFIG_DIR / "chunk_config.yaml"))

    # 청킹 결과 확인
    chunk_files = list(CHUNK_PROJECT_DIR.glob("**/*.parquet"))
    if chunk_files:
        print(f"[SUCCESS] 청킹 완료: {chunk_files[0]}")
        return chunk_files[0]
    else:
        print("[ERROR] 청킹 결과 파일이 생성되지 않았습니다.")
        sys.exit(1)


def validate_corpus(corpus_path: Path):
    """코퍼스 데이터 검증"""
    import pandas as pd

    print("\n" + "=" * 60)
    print("코퍼스 데이터 검증")
    print("=" * 60)

    df = pd.read_parquet(corpus_path)
    print(f"  총 청크 수: {len(df)}")
    print(f"  컬럼: {list(df.columns)}")
    print(f"  doc_id 유니크 수: {df['doc_id'].nunique()}")
    print(f"  평균 내용 길이: {df['contents'].str.len().mean():.0f} chars")
    print(f"\n  첫 번째 청크 미리보기:")
    print(f"  {df['contents'].iloc[0][:200]}...")

    # data/ 디렉토리로 복사
    output_path = BENCHMARK_DIR / "data" / "corpus.parquet"
    df.to_parquet(output_path)
    print(f"\n[SUCCESS] corpus.parquet 저장: {output_path}")
    return output_path


if __name__ == "__main__":
    print("AutoRAG 벤치마크 - PDF 파싱 및 청킹")
    print("=" * 60)

    # PDF 파일 확인
    check_pdf_files()

    # Stage 1: 파싱
    parsed_path = run_parsing()

    # Stage 2: 청킹
    corpus_path = run_chunking(parsed_path)

    # 검증 및 저장
    validate_corpus(corpus_path)

    print("\n" + "=" * 60)
    print("다음 단계: python scripts/02_create_qa_dataset.py")
    print("=" * 60)
