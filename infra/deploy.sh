#!/usr/bin/env bash
# Esqueleto AWS — completar ACCOUNT_ID, REGION, REPO
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:?Define AWS_ACCOUNT_ID}"
REPO="plant-disease-lambda"
IMAGE_TAG="latest"

echo "1. Build Docker (desde lambda/ con model.onnx y model.meta.json copiados)"
echo "   cp artifacts/model.onnx lambda/"
echo "   cp artifacts/model.meta.json lambda/"
echo "   cd lambda && docker build -t ${REPO}:${IMAGE_TAG} ."

echo "2. ECR login + push"
# aws ecr create-repository --repository-name ${REPO} || true
# docker tag ${REPO}:${IMAGE_TAG} ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${IMAGE_TAG}
# aws ecr get-login-password --region ${REGION} | docker login ...
# docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${IMAGE_TAG}

echo "3. Crear Lambda (container) + API Gateway HTTP API con CORS"
echo "   Ver documentación AWS Lambda container images."
