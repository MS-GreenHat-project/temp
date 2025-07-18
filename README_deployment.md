# 쿠버네티스 배포 가이드

이 가이드는 YOLO 모델을 쿠버네티스에 배포하기 위한 CI/CD 파이프라인 설정을 설명합니다.

## 🏗️ 아키텍처

```
GitHub Repository
    ↓
GitHub Actions (CI/CD)
    ↓
Container Registry (GHCR)
    ↓
ArgoCD (GitOps)
    ↓
Kubernetes Cluster
```

## 📋 사전 요구사항

### 1. 쿠버네티스 클러스터
- **AKS (Azure Kubernetes Service)** 또는 **EKS (Amazon EKS)**
- **GPU 노드** 지원 (NVIDIA GPU)
- **Ingress Controller** (NGINX)
- **Storage Class** 설정

### 2. ArgoCD 설치
```bash
# ArgoCD 설치
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# ArgoCD CLI 설치
brew install argocd  # macOS
# 또는
curl -sSL -o argocd-linux-amd64 https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
sudo install -m 555 argocd-linux-amd64 /usr/local/bin/argocd
```

### 3. GitHub Secrets 설정
```bash
# GitHub Repository Settings > Secrets and variables > Actions
KUBE_CONFIG_STAGING=<base64-encoded-kubeconfig>
KUBE_CONFIG_PRODUCTION=<base64-encoded-kubeconfig>
SLACK_WEBHOOK_URL=<slack-webhook-url>
```

## 🚀 배포 프로세스

### 1. 모델 훈련 및 평가
```bash
# Azure ML 파이프라인 실행
az ml job create --file pipeline.yaml
```

### 2. 배포 매니페스트 생성
```bash
python deploy.py \
    --model-folder /path/to/model \
    --eval-folder /path/to/evaluation \
    --namespace greenhat-ai \
    --image-repo ghcr.io/your-org/greenhat-ai \
    --image-tag latest \
    --replicas 2
```

### 3. GitHub Actions 워크플로우 실행
1. **GitHub Repository** → **Actions** 탭
2. **Deploy to Kubernetes** 워크플로우 선택
3. **Run workflow** 클릭
4. 파라미터 입력:
   - Model version: `20250718-021239`
   - Image tag: `latest`
   - Namespace: `greenhat-ai`
   - Replicas: `2`
   - Environment: `staging`

### 4. ArgoCD 자동 배포
- GitHub Actions가 매니페스트를 업데이트
- ArgoCD가 자동으로 변경사항 감지
- 쿠버네티스에 자동 배포

## 📁 생성되는 파일들

### 1. 쿠버네티스 매니페스트
```
k8s-manifests/
├── configmap.yaml          # 모델 설정
├── deployment.yaml         # 애플리케이션 배포
├── service.yaml           # 서비스 노출
├── ingress.yaml           # 인그레스 설정
├── pvc.yaml              # 영구 볼륨 클레임
└── argocd-application.yaml # ArgoCD 애플리케이션
```

### 2. GitHub Actions 워크플로우
```
.github/workflows/
└── deploy-to-k8s.yml      # CI/CD 파이프라인
```

## 🔧 설정 파일

### 1. Dockerfile
```dockerfile
# ml-pipline/Dockerfile
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
# ... (기존 내용)
```

### 2. ArgoCD Application
```yaml
# argocd-app-example.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: greenhat-ai
  namespace: argocd
# ... (기존 내용)
```

## 🎯 배포 확인

### 1. ArgoCD UI 확인
```bash
# ArgoCD 서비스 포트포워딩
kubectl port-forward svc/argocd-server -n argocd 8080:443

# 브라우저에서 접속
# https://localhost:8080
# Username: admin
# Password: kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

### 2. 쿠버네티스 리소스 확인
```bash
# 파드 상태 확인
kubectl get pods -n greenhat-ai

# 서비스 확인
kubectl get svc -n greenhat-ai

# 인그레스 확인
kubectl get ingress -n greenhat-ai

# 로그 확인
kubectl logs -f deployment/greenhat-ai-deployment -n greenhat-ai
```

### 3. 헬스체크
```bash
# 서비스 헬스체크
curl http://greenhat-ai-service.greenhat-ai.svc.cluster.local/health

# 인그레스 헬스체크 (외부 접근)
curl http://greenhat-ai.your-domain.com/health
```

## 🔄 롤백 프로세스

### 1. ArgoCD를 통한 롤백
```bash
# 이전 버전으로 롤백
argocd app rollback greenhat-ai

# 특정 리비전으로 롤백
argocd app rollback greenhat-ai <revision>
```

### 2. 쿠버네티스 직접 롤백
```bash
# 이전 배포로 롤백
kubectl rollout undo deployment/greenhat-ai-deployment -n greenhat-ai

# 특정 리비전으로 롤백
kubectl rollout undo deployment/greenhat-ai-deployment -n greenhat-ai --to-revision=2
```

## 📊 모니터링

### 1. Prometheus + Grafana
```yaml
# 모니터링 설정 예시
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: greenhat-ai-monitor
  namespace: greenhat-ai
spec:
  selector:
    matchLabels:
      app: greenhat-ai
  endpoints:
  - port: http
    path: /metrics
```

### 2. 로그 수집
```yaml
# Fluentd 설정 예시
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/greenhat-ai-*.log
      pos_file /var/log/greenhat-ai.log.pos
      tag greenhat-ai
      read_from_head true
      <parse>
        @type json
        time_key time
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>
```

## 🚨 문제 해결

### 1. 일반적인 문제들

#### 파드가 시작되지 않음
```bash
# 파드 상태 확인
kubectl describe pod <pod-name> -n greenhat-ai

# 로그 확인
kubectl logs <pod-name> -n greenhat-ai
```

#### 이미지 풀 에러
```bash
# 이미지 풀 시크릿 확인
kubectl get secrets -n greenhat-ai

# 이미지 풀 시크릿 생성
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=<github-username> \
  --docker-password=<github-token> \
  --namespace=greenhat-ai
```

#### GPU 할당 문제
```bash
# GPU 노드 확인
kubectl get nodes -o json | jq '.items[] | select(.status.allocatable."nvidia.com/gpu" != null)'

# GPU 드라이버 확인
kubectl get pods -n kube-system | grep nvidia
```

### 2. 디버깅 명령어
```bash
# 파드 내부 접속
kubectl exec -it <pod-name> -n greenhat-ai -- /bin/bash

# 환경 변수 확인
kubectl exec <pod-name> -n greenhat-ai -- env

# 설정 파일 확인
kubectl exec <pod-name> -n greenhat-ai -- cat /app/config/config.json
```

## 🔐 보안 설정

### 1. RBAC 설정
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: greenhat-ai
  name: greenhat-ai-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
```

### 2. 네트워크 정책
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: greenhat-ai-network-policy
  namespace: greenhat-ai
spec:
  podSelector:
    matchLabels:
      app: greenhat-ai
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
```

## 📈 성능 최적화

### 1. 리소스 설정
```yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
    nvidia.com/gpu: "1"
```

### 2. HPA (Horizontal Pod Autoscaler)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: greenhat-ai-hpa
  namespace: greenhat-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: greenhat-ai-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 🎉 배포 완료 확인

배포가 성공적으로 완료되면 다음을 확인하세요:

1. ✅ **ArgoCD UI**에서 애플리케이션 상태가 "Healthy"
2. ✅ **쿠버네티스 파드**가 "Running" 상태
3. ✅ **서비스 엔드포인트** 응답 확인
4. ✅ **모니터링 메트릭** 수집 확인
5. ✅ **로드 테스트** 통과 확인

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. ArgoCD 애플리케이션 로그
2. 쿠버네티스 이벤트
3. 파드 로그
4. 네트워크 연결 상태
5. 리소스 사용량 