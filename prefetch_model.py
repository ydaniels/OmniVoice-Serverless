#!/usr/bin/env python3
from __future__ import annotations

import os

from huggingface_hub import snapshot_download


MODEL_ID = os.getenv("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
AUDIO_TOKENIZER_ID = os.getenv(
    "OMNIVOICE_AUDIO_TOKENIZER_ID", "eustlb/higgs-audio-v2-tokenizer"
)
ASR_MODEL_ID = os.getenv("OMNIVOICE_ASR_MODEL_ID", "openai/whisper-large-v3-turbo")
DOWNLOAD_ASR = (
    os.getenv("OMNIVOICE_DOWNLOAD_ASR_AT_BUILD", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)


def main() -> int:
    print(f"Prefetching OmniVoice model: {MODEL_ID}")
    snapshot_download(MODEL_ID, resume_download=True)

    print(f"Prefetching OmniVoice audio tokenizer: {AUDIO_TOKENIZER_ID}")
    snapshot_download(AUDIO_TOKENIZER_ID, resume_download=True)

    if DOWNLOAD_ASR:
        print(f"Prefetching Whisper ASR model: {ASR_MODEL_ID}")
        snapshot_download(ASR_MODEL_ID, resume_download=True)

    print("Model prefetch completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
