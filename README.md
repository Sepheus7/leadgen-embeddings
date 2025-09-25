# LeadGen Embeddings (Day 1)

Minimal prototype for lead scoring using text + tabular embeddings and FAISS.

## Quickstart

```bash
make data
make indices
make api
# In another shell
curl -X POST localhost:8000/score_lead -H "Content-Type: application/json" -d '{
  "customer_id": 0,
  "name": "Jane Doe",
  "industry": "Finance",
  "company_size": 500,
  "country": "US",
  "job_title": "Portfolio Manager",
  "bio": "Works on trading strategies and allocator relationships.",
  "web_activity_score": 0.8,
  "email_engagement_score": 0.7
}'
```

## Architecture (ASCII)

```
        +----------------+        +---------------------+        +------------------+
        |  CRM Parquet   |        |  Preprocess & Feat  |        |  Embeddings      |
        |  (data/*.pq)   |  -->   |  - text blob        |  -->   |  - text (E5)     |
        +----------------+        |  - freq encoders    |        |  - tabular (PCA) |
                                   +---------------------+        +------------------+
                                                |                           |
                                                v                           v
                                          +-----------+              +-------------+
                                          |  Concate  |  ----->      |  L2 Norm    |
                                          +-----------+              +-------------+
                                                  |                          |
                                                  v                          v
                                        +-------------------+     +-------------------+
                                        | FAISS Index (all) |     | FAISS Index (high)|
                                        +-------------------+     +-------------------+
                                                     |                     |
                                                     v                     v
                                              +-------------------------------+
                                              | FastAPI: POST /score_lead     |
                                              | Returns: S_look, S_novel,     |
                                              | contrast, neighbor ids        |
                                              +-------------------------------+
```

## Why FAISS by default?

- Simple, fast ANN with cosine via inner product on normalized vectors.
- No infra dependency for Day-1. pgvector is optional (see `infra/docker-compose.yml`).

## Notes

- k=20 hardcoded for Day-1; thresholds TBD in Day-2 notebooks.
- Explanations are placeholders (nearest neighbor ids); richer explanations to come.

## Deployment on AWS

### Artifacts and image

- Bake artifacts into the image for fastest startup (recommended):
  - During Docker build, run:
    ```bash
    make data && PYTHONPATH=. make indices
    ```
  - This produces `artifacts/` (FAISS indices, scaler/PCA, feature_meta, emails) inside the image.
- Offline mode works (no Hugging Face). The text embedder falls back to a hashing vectorizer.
- Container listens on port 8000 with health at `/health`.

### Option A: App Runner (simplest managed)

- Push the image to ECR.
- Create App Runner service from ECR image:
  - Port: 8000
  - Health check path: `/health`
  - CPU/memory: start with 1 vCPU / 2 GB
  - Auto scaling: min 1, max per traffic
- Pros: zero infra to manage, TLS/URL out of the box. Cons: fewer networking knobs than ECS.

### Option B: ECS Fargate (more control)

- Push image to ECR. Create ECS cluster and service (Fargate):
  - Task definition: container port 8000
  - Service: attach an ALB, target group health path `/health`
  - CPU/memory: e.g., 0.5–1 vCPU / 1–2 GB
  - Desired count: >=1; add autoscaling policies as needed
- Pros: VPC control, scaling policies, enterprise fit. Cons: more setup than App Runner.

### Option C: Lambda (serverless, cold-start tradeoffs)

- Package as a container image; add a lightweight adapter (e.g., `mangum`) to run FastAPI on Lambda/ALB or API Gateway.
- Provisioned concurrency recommended to reduce cold starts (FAISS/model load time). Memory 2048–4096 MB is typical.
- Pros: pay-per-use, zero servers. Cons: cold starts, request/response size/time limits.

### Build and push (ECR)

```bash
AWS_ACCOUNT=123456789012
AWS_REGION=eu-west-1
REPO=leadgen-embeddings

aws ecr create-repository --repository-name $REPO || true
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t $REPO .
docker tag $REPO:latest $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
docker push $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
```

### Runtime configuration

- No required env vars for offline mode.
- If using a local SentenceTransformer model, point `TEXT_MODEL_NAME_PRIMARY` in `leadgen/config.py` to a local path baked in the image.
- Health: `GET /health`
- Scoring: `POST /score_lead`

### Choosing a platform

- App Runner: choose for fastest path to a managed HTTPS API.
- ECS Fargate: choose for VPC integration and fine-grained scaling.
- Lambda: choose for bursty workloads with provisioned concurrency to tame cold starts.

