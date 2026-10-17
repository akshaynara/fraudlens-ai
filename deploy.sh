#!/usr/bin/env bash
# One-shot deploy to Cloud Run from source (Cloud Build does the Docker build).
#
# Why these flags:
#   --max-instances 1      cases live in memory; with >1 instance the UI's polling can hit an
#                          instance that never saw the case (404).
#   --min-instances 1      keeps the instance warm so judges never hit a cold start.
#   --no-cpu-throttling    the investigation runs in a background thread after the POST returns;
#                          without this Cloud Run throttles CPU outside requests and the agent stalls.
set -euo pipefail
PROJECT="${PROJECT:-$(gcloud config get-value project)}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-fraudlens}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY (or edit this script to use Vertex AI instead)}"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  firestore.googleapis.com --project "$PROJECT"

# Firestore (Native mode) for the case archive. Safe to re-run.
if ! gcloud firestore databases describe --database="(default)" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud firestore databases create --database="(default)" --location="$REGION" --type=firestore-native --project "$PROJECT"
fi
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:${RUNTIME_SA}" \
  --role=roles/datastore.user --condition=None --quiet >/dev/null

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
