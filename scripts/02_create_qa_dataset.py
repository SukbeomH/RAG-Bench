"""
02. QA 데이터셋 생성 스크립트

청킹된 코퍼스로부터 OpenAI GPT-4o-mini를 사용하여 한국어 QA 쌍을 자동 생성한다.
OPENAI_API_KEY 환경변수가 설정되어 있어야 한다.
"""
import os
import sys
from pathlib import Path

import pandas as pd

# SSL 인증서 설정 (선택 - 사설 CA 사용 시 SSL_CERT_FILE 환경변수 설정)
CERT_PATH = os.getenv("SSL_CERT_FILE")
if CERT_PATH and Path(CERT_PATH).exists():
    os.environ["SSL_CERT_FILE"] = CERT_PATH
    os.environ["REQUESTS_CA_BUNDLE"] = CERT_PATH
    os.environ["CURL_CA_BUNDLE"] = CERT_PATH

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "autorag_benchmark"
DATA_DIR = BENCHMARK_DIR / "data"
CHUNK_PROJECT_DIR = BENCHMARK_DIR / "chunk_project"
PARSE_PROJECT_DIR = BENCHMARK_DIR / "parse_project"

# QA 생성 설정
NUM_SAMPLES = 100  # 생성할 QA 쌍 수
LLM_MODEL = "gpt-4o-mini"


def check_environment():
    """환경 변수 및 데이터 확인"""
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    corpus_path = find_corpus_parquet()
    if corpus_path is None:
        print("[ERROR] corpus.parquet 파일을 찾을 수 없습니다.")
        print("먼저 scripts/01_parse_and_chunk.py를 실행해주세요.")
        sys.exit(1)

    parsed_path = find_parsed_parquet()
    if parsed_path is None:
        print("[ERROR] parsed.parquet 파일을 찾을 수 없습니다.")
        print("먼저 scripts/01_parse_and_chunk.py를 실행해주세요.")
        sys.exit(1)

    return parsed_path, corpus_path


def find_corpus_parquet():
    """청킹된 corpus parquet 파일 찾기"""
    # data/ 에 이미 있으면 사용
    if (DATA_DIR / "corpus.parquet").exists():
        return DATA_DIR / "corpus.parquet"
    # chunk_project 에서 찾기
    files = list(CHUNK_PROJECT_DIR.glob("**/*corpus*.parquet"))
    if not files:
        files = list(CHUNK_PROJECT_DIR.glob("**/*.parquet"))
    return files[0] if files else None


def find_parsed_parquet():
    """파싱된 parquet 파일 찾기"""
    files = list(PARSE_PROJECT_DIR.glob("**/*.parquet"))
    return files[0] if files else None


def create_qa_dataset(parsed_path: Path, corpus_path: Path):
    """QA 데이터셋 생성"""
    from llama_index.llms.openai import OpenAI

    from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
    from autorag.data.qa.generation_gt.llama_index_gen_gt import (
        make_basic_gen_gt,
        make_concise_gen_gt,
    )
    from autorag.data.qa.query.llama_gen_query import factoid_query_gen
    from autorag.data.qa.sample import random_single_hop
    from autorag.data.qa.schema import Corpus, Raw

    print("\n" + "=" * 60)
    print("QA 데이터셋 생성")
    print(f"  LLM: {LLM_MODEL}")
    print(f"  목표 샘플 수: {NUM_SAMPLES}")
    print("=" * 60)

    # LLM 초기화
    llm = OpenAI(model=LLM_MODEL, temperature=0.7)

    # 데이터 로드
    raw_df = pd.read_parquet(parsed_path)
    corpus_df = pd.read_parquet(corpus_path)

    print(f"  파싱 문서 수: {len(raw_df)}")
    print(f"  코퍼스 청크 수: {len(corpus_df)}")

    raw_instance = Raw(raw_df)
    corpus_instance = Corpus(corpus_df, raw_instance)

    # 샘플 수 조정 (코퍼스 크기에 맞춤)
    sample_n = min(NUM_SAMPLES, len(corpus_df))
    print(f"  실제 샘플 수: {sample_n}")

    # QA 생성 파이프라인
    print("\n[1/4] 랜덤 샘플링...")
    qa = corpus_instance.sample(random_single_hop, n=sample_n)
    qa = qa.map(lambda df: df.reset_index(drop=True))
    qa = qa.make_retrieval_gt_contents()

    print("[2/4] 질문 생성 (factoid query)...")
    qa = qa.batch_apply(factoid_query_gen, llm=llm)

    print("[3/4] 정답 생성 (basic + concise)...")
    qa = qa.batch_apply(make_basic_gen_gt, llm=llm)
    qa = qa.batch_apply(make_concise_gen_gt, llm=llm)

    print("[4/4] 필터링 (dontknow)...")
    qa = qa.filter(dontknow_filter_rule_based, lang="ko")

    return qa


def save_and_validate(qa):
    """QA 데이터셋 저장 및 검증"""
    print("\n" + "=" * 60)
    print("QA 데이터셋 저장")
    print("=" * 60)

    qa_path = DATA_DIR / "qa.parquet"
    corpus_path = DATA_DIR / "corpus.parquet"

    qa.to_parquet(str(qa_path), str(corpus_path))

    # 검증
    qa_df = pd.read_parquet(qa_path)
    corpus_df = pd.read_parquet(corpus_path)

    print(f"\n  QA 데이터셋:")
    print(f"    총 QA 쌍: {len(qa_df)}")
    print(f"    컬럼: {list(qa_df.columns)}")

    print(f"\n  코퍼스 데이터셋:")
    print(f"    총 청크: {len(corpus_df)}")
    print(f"    컬럼: {list(corpus_df.columns)}")

    print(f"\n  샘플 QA:")
    for i, row in qa_df.head(3).iterrows():
        print(f"    Q: {row['query'][:80]}...")
        if "generation_gt" in row:
            gt = row["generation_gt"]
            if hasattr(gt, '__len__') and len(gt) > 0:
                gt = gt[0]
            print(f"    A: {str(gt)[:80]}...")
        print()

    print(f"[SUCCESS] 저장 완료:")
    print(f"  - {qa_path}")
    print(f"  - {corpus_path}")

    return qa_path, corpus_path


if __name__ == "__main__":
    print("AutoRAG 벤치마크 - QA 데이터셋 생성")
    print("=" * 60)

    # 환경 확인
    parsed_path, corpus_path = check_environment()

    # QA 생성
    qa = create_qa_dataset(parsed_path, corpus_path)

    # 저장 및 검증
    save_and_validate(qa)

    print("\n" + "=" * 60)
    print("다음 단계: python scripts/03_run_benchmark.py")
    print("=" * 60)
