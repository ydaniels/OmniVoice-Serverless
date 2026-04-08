# OmniVoice RunPod Serverless

This repository contains a RunPod Serverless worker for OmniVoice.

## What it does

- Runs OmniVoice via RunPod Serverless handler (`handler.py`).
- Installs OmniVoice from GitHub during image build.
- Prefetches model artifacts at build time so runtime avoids first-request downloads.
- Optionally uploads generated audio to S3 when credentials + bucket are present in env.

## Build image locally

```bash
docker build -f Dockerfile -t omni-serverless:local .
```

## Run container locally (quick smoke)

```bash
docker run --rm --gpus all -e LOG_LEVEL=INFO omni-serverless:local
```

## RunPod payload examples

### Auto voice

```json
{
  "input": {
    "text": "Hello from OmniVoice serverless."
  }
}
```

### Voice design

```json
{
  "input": {
    "text": "This is a multilingual voice design test.",
    "language": "English",
    "instruct": "female, low pitch, british accent"
  }
}
```

### Voice cloning (URL reference audio)

```json
{
  "input": {
    "text": "This sentence is generated with voice cloning.",
    "ref_audio": "https://example.com/reference.wav",
    "ref_text": "This is the transcript of the reference audio."
  }
}
```

### Voice cloning (base64 reference audio)

```json
{
  "input": {
    "text": "Voice cloning from base64 input.",
    "ref_audio_base64": "<base64-wav-bytes>"
  }
}
```

## Response shape

```json
{
  "format": "wav",
  "sample_rate": 24000,
  "duration_seconds": 3.214,
  "elapsed_seconds": 2.781,
  "model_id": "k2-fsa/OmniVoice",
  "device": "cuda:0",
  "s3_bucket": "my-bucket",
  "s3_key": "omnivoice/outputs/2026/04/08/....wav",
  "s3_url": "https://...",
  "public_url": "https://render.example.com/omnivoice/outputs/2026/04/08/....wav"
}
```

`s3_*` fields are only returned when S3 upload is enabled and successful.  
When S3 upload succeeds, `audio_base64` is omitted.

## S3 upload env vars

Required to enable upload:

- `S3_ACCESS_KEY_ID` or `AWS_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY` or `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET`

Optional:

- `S3_SESSION_TOKEN` or `AWS_SESSION_TOKEN`
- `AWS_REGION` or `S3_REGION`
- `S3_ENDPOINT_URL` (for S3-compatible stores)
- `S3_PREFIX` (default: `omnivoice/outputs`)
- `S3_CUSTOM_DOMAIN` (preferred, used to build returned public URL)
- `S3_PUBLIC_BASE_URL` (legacy fallback)
- `S3_OBJECT_ACL`

## Optional input flags

- `return_base64` (default `true`, but ignored when S3 upload succeeds)
- `upload_to_s3` (default `true` when S3 env is configured)
