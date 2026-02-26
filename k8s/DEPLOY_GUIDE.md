# K8s 벤치마크 배포 가이드

## 환경

| 항목 | 값 |
|------|------|
| 클러스터 | `zcp-ags-cp-eks` (EKS, ap-northeast-2) |
| 네임스페이스 | `rag-bench-test` |
| 레지스트리 | `$HARBOR_REGISTRY` 환경변수 (Harbor) |
| PVC | EFS (`efs-zcp`, ReadWriteMany) |
| 스케줄링 | 5노드 활용 — management 3대(m7i/m8i.2xlarge) + ai-platform 2대(r5a.xlarge, toleration 적용) |
| 시크릿 소스 | `.env` 파일 (OPENAI_API_KEY, HARBOR_REGISTRY, HARBOR_USER, HARBOR_CLI_SECRET) |

---

## Step 1: Harbor 프로젝트 생성

Harbor UI(`https://$HARBOR_REGISTRY`)에서 `rag-bench-test` 프로젝트 생성.
이미 존재하면 생략.

## Step 2: 이미지 빌드 & 푸시

### 방법 A: K8s 원격 빌더 (권장)

K8s management 노드에서 네이티브 amd64로 빌드 + 직접 푸시. 로컬 에뮬레이션 대비 **3~5배 빠름**.

```bash
cd /Users/sukbeom/Desktop/autorag
eval "$(grep -E '^HARBOR' .env)"

# 빌더 생성 + 부트스트랩 (최초 1회)
docker buildx create --name k8s-amd64 --driver kubernetes \
    --driver-opt "namespace=rag-bench-test,nodeselector=node-role.kubernetes.io/management=management" \
    --platform linux/amd64
docker buildx inspect k8s-amd64 --bootstrap

# 빌드 + 레지스트리 직접 푸시
docker buildx build --builder k8s-amd64 --platform linux/amd64 --push \
    -t $HARBOR_REGISTRY/rag-bench-test/worker:latest \
    -f k8s/Dockerfile .

# 빌드 완료 후 빌더 제거 (클러스터 리소스 해제)
docker buildx rm k8s-amd64
```

> `--push` 필수: 원격 빌더는 `--load` 미지원. Harbor 인증은 `harbor-cred` Secret이 아닌 로컬 docker login 사용.

### 방법 B: 로컬 빌드 (대안)

Apple Silicon에서 에뮬레이션 빌드. 느리지만 K8s 접근 없이 가능.

```bash
cd /Users/sukbeom/Desktop/autorag
eval "$(grep -E '^HARBOR' .env)"

# Harbor 로그인
echo $HARBOR_CLI_SECRET | docker login "$HARBOR_REGISTRY" -u "$HARBOR_USER" --password-stdin

# 빌드 (amd64 에뮬레이션)
docker buildx build --platform linux/amd64 \
    -t $HARBOR_REGISTRY/rag-bench-test/worker:latest \
    -f k8s/Dockerfile --load .

# 푸시
docker push $HARBOR_REGISTRY/rag-bench-test/worker:latest
```

## Step 3: 인프라 리소스 생성

```bash
# 네임스페이스 + PVC
kubectl apply -f k8s/manifests/namespace.yaml
kubectl apply -f k8s/manifests/results-pvc.yaml
kubectl apply -f k8s/manifests/model-cache-pvc.yaml

# PVC Bound 확인
kubectl get pvc -n rag-bench-test
```

## Step 4: Secret 생성

```bash
# .env에서 키 로드
source <(grep -E 'OPENAI_API_KEY|HARBOR_REGISTRY|HARBOR_CLI_SECRET|HARBOR_USER' .env)

# OpenAI API Key
kubectl create secret generic bench-secrets \
    -n rag-bench-test \
    --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY"

# Harbor ImagePullSecret
kubectl create secret docker-registry harbor-cred \
    -n rag-bench-test \
    --docker-server=$HARBOR_REGISTRY \
    --docker-username="$HARBOR_USER" \
    --docker-password="$HARBOR_CLI_SECRET"

# 확인
kubectl get secrets -n rag-bench-test
```

## Step 5: dry-run 확인

```bash
source <(grep HARBOR_REGISTRY .env)
export IMAGE=$HARBOR_REGISTRY/rag-bench-test/worker:latest

# 전체 YAML 확인
python k8s/orchestrator.py --image $IMAGE --dry-run

# 단일 카테고리만
python k8s/orchestrator.py --image $IMAGE --dry-run --categories general
```

## Step 6: 실행

```bash
source <(grep -E 'HARBOR_REGISTRY|OPENAI_API_KEY' .env)
export IMAGE=$HARBOR_REGISTRY/rag-bench-test/worker:latest

# 전체 실행 (4카테고리 × 6조합 = 28 Jobs)
python k8s/orchestrator.py --image $IMAGE

# 특정 카테고리만
python k8s/orchestrator.py --image $IMAGE --categories general,legal

# 데이터 크기 제한 (테스트용)
python k8s/orchestrator.py --image $IMAGE --categories general \
    --max-corpus 1000 --max-queries 50
```

> 기본 리소스: Prep CPU 1/1, Mem 4Gi/8Gi | Bench CPU 1/2, Mem 4Gi/8Gi.
> ai-platform toleration 적용으로 5노드(~13 vCPU) 활용 가능.

오케스트레이터가 자동으로:
1. Phase 1 (prep) 4개 Job 생성 → 완료 대기
2. PVC 가시성 검증
3. Phase 2 (bench) 24개 Job 생성 → 완료 대기
4. 결과 수집 + 병합

## Step 7: 모니터링

별도 터미널에서:

```bash
# Job 상태 확인
kubectl get jobs -n rag-bench-test -l phase=prep
kubectl get jobs -n rag-bench-test -l phase=bench

# Pod 상태
kubectl get pods -n rag-bench-test --sort-by=.metadata.creationTimestamp

# 특정 Job 로그
kubectl logs job/prep-general-<run-id> -n rag-bench-test -c worker -f
kubectl logs job/bench-general-bgem3-koreanbm25-colbert-<run-id> -n rag-bench-test -c worker -f
```

## Step 8: 결과 확인

```bash
# 오케스트레이터가 자동 수집하지만, 수동 수집도 가능
python k8s/orchestrator.py --image $IMAGE --collect-only --run-id <RUN_ID>

# 로컬 결과
ls k8s_results/<RUN_ID>/
# general/  legal/  business/  medical/  merged_report.html
```

---

## 문제 해결

### Harbor 인증 실패 (이미지 Pull 불가)

```bash
# Secret 확인
kubectl get secret harbor-cred -n rag-bench-test -o yaml

# Pod 이벤트 확인
kubectl describe pod <pod-name> -n rag-bench-test | tail -20
```

### Phase 1 실패 시 재실행

```bash
# 실패 Job 삭제 후 재실행
kubectl delete job prep-general-<run-id> -n rag-bench-test
python k8s/orchestrator.py --image $IMAGE --categories general
```

### Phase 2만 재실행 (Phase 1 결과 재사용)

```bash
python k8s/orchestrator.py --image $IMAGE --skip-prep --run-id <이전_RUN_ID>
```

### 리소스 부족 (Pending Pods)

5노드(~13 vCPU)에서 동시 24개 Bench Job은 Pending 발생 가능.
오케스트레이터가 자동으로 대기/순차 실행함.

```bash
# 방법 1: 카테고리 나눠 실행
python k8s/orchestrator.py --image $IMAGE --categories general,legal
python k8s/orchestrator.py --image $IMAGE --categories business,medical

# 방법 2: 리소스 추가 축소
python k8s/orchestrator.py --image $IMAGE \
    --bench-cpu-limit 1  # ColBERT 로딩 느려짐 주의
```

### K8s 빌더 리소스 충돌

`docker buildx create --driver kubernetes`로 생성한 빌더 Pod가 클러스터 리소스를 점유한다.
**빌드 후 반드시 제거**:

```bash
docker buildx rm k8s-amd64
```

### 전체 정리

```bash
# 특정 실행 정리
python k8s/orchestrator.py --image $IMAGE --cleanup --run-id <RUN_ID>

# 네임스페이스 전체 삭제
kubectl delete ns rag-bench-test
```
