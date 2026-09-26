# models/ — our trained artifacts for the Docker image

The Docker build copies this folder to `/app/models` (the image's `HEARSAY_MODELS`). It holds only what
**we** trained, plus the chosen fusion. The public pretrained weights (XLS-R/AASIST and XLS-R/SLS ONNX
exports, WavLM-Large, the XLS-R config, the SpeechBrain speaker model) are not stored here. The build
downloads them from pinned Hugging Face commits (`scripts/fetch_models.sh`, `scripts/public_models.tsv`).

Fill it on the machine where the models were trained:

```bash
scripts/export_models.sh --version v2h        # submissions/v2h/fusion_meta.json + what that fusion needs
```

| File | What it is |
|---|---|
| `fusion_meta.json` | the fusion: which detectors, dev-set z-scoring, weights, Platt calibration, prior shift |
| `wavlm_head.npz` | logistic regression on frozen WavLM-Large layer means (layers, scaler, coefficients) |
| `forensic_gbm.txt`, `forensic_gbm_cols.json` | glass-box LightGBM over ~200 named forensic features |
| `tak_ft/best.pt`, `tak_head/best.pt` | our fine-tuned XLS-R + AASIST checkpoints, stored as float16 (only if the fusion uses them) |
| `selftrain_base.npz` | base features for transductive self-training (only if the fusion uses `wavlm_selftrain`) |
| `test_transcripts.json` | cached Whisper transcripts of the test clips, used only for report text |
| `MANIFEST.json`, `SHA256SUMS` | provenance, and checksums that the Docker build verifies |

`.pt` and `.npz` files are git-ignored. To make a fresh clone buildable, either force-add the small ones
(`git add -f models/wavlm_head.npz`) or publish the archive from `scripts/export_models.sh --tar` and build
with `--build-arg OWN_MODELS_URL=...`. See `docs/DOCKER.md`.
