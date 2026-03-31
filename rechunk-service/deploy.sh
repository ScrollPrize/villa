#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="rechunk-service"
TAG="${TAG:-latest}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="${ECR_REPO:-}"
PORT="${PORT:-8000}"

cd "$(dirname "$0")"

echo "Building $IMAGE_NAME:$TAG"
docker build -t "$IMAGE_NAME:$TAG" .

if [ -n "$ECR_REPO" ]; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO"

    echo "Logging in to ECR"
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ECR_URI"

    echo "Pushing to $ECR_URI:$TAG"
    docker tag "$IMAGE_NAME:$TAG" "$ECR_URI:$TAG"
    docker push "$ECR_URI:$TAG"
    echo "Pushed: $ECR_URI:$TAG"
else
    echo "Running locally on port $PORT"
    echo "Pass AWS credentials via environment or instance profile."
    docker run --rm -it \
        -p "$PORT:8000" \
        -e AWS_ACCESS_KEY_ID \
        -e AWS_SECRET_ACCESS_KEY \
        -e AWS_SESSION_TOKEN \
        -e AWS_DEFAULT_REGION="$AWS_REGION" \
        "$IMAGE_NAME:$TAG"
fi
