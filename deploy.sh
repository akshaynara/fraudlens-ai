#!/usr/bin/env bash
# One-shot deploy to Cloud Run from source (Cloud Build does the Docker build).
#
# Why these flags:
#   --max-instances 1      cases live in memory; with >1 instance the UI's polling can hit an
#                          instance that never saw the case (404).
#   --min-instances 1      keeps the instance warm so judges never hit a cold start, and live
#                          cases survive between requests.
#   --no-cpu-throttling    the investigation runs in a background thread after the POST returns;
#                          without this Cloud Run throttles CPU outside requests and the agent stalls.
set -euo pipefail
PROJECT="${PROJECT:-$(gcloud config get-value project)}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-fraudlens}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY (or edit this script to use Vertex AI instead)}"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project "$PROJECT"
gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" --region "$REGION" \
  --allow-unauthenticated \
  --memory 1Gi --cpu 1 --timeout 300 \
  --min-instances 1 --max-instances 1 --no-cpu-throttling \
  --set-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY},GEMINI_MODEL=${GEMINI_MODEL:-gemini-3.8-flash},GEMINI_MODEL_PRO=${GEMINI_MODEL_PRO:-}"

echo
echo "Deployed. URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format 'value(status.url)'
