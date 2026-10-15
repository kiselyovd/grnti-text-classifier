---
title: GRNTI Classifier
emoji: 📚
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
models:
  - kiselyovd/grnti-text-classifier
short_description: Russian scientific abstracts → five GRNTI subjects
---

# GRNTI Classifier

Real XLM-RoBERTa inference on Russian scientific text: five exact codes, full subject names and uncalibrated model scores. English/Russian interface. Runs on CPU in the free HF Space by default; ordinary requests do not allocate GPU time.

Weights: `kiselyovd/grnti-text-classifier@24fbe75c27112612a464935c95465def6af788ef`, SHA256 `324f0ca0b15fd30f8b0c9bd2e9db4594b88119489f3a8d1df0169a455ca1eba4`. Safetensors only; no remote model code. Input: at most 20,000 characters; first 256 tokens; 28 checkpoint labels. Five- and six-digit codes are preserved exactly; 680000 and 683500 intentionally retain their duplicate stored names.

Illustrative examples are author-written, not benchmark ground truth. Scores are not calibrated confidence estimates. No user text is logged or cached by this application. Processing takes place on this public HF Space; avoid confidential inputs.

[Model card](https://huggingface.co/kiselyovd/grnti-text-classifier) · [Source, evaluation and limitations](https://github.com/kiselyovd/grnti-text-classifier)

Local production entry: `python app.py` after installing `requirements.txt`. CPU is the default locally and on HF; model files reuse the normal HF cache. Code MIT; checkpoint's declared license MIT, XLM-R upstream MIT; all upstream package licenses retain their terms.

## Execution mode

CPU is the default for both local runs and the free hosted demo. Ordinary requests do not allocate GPU time or consume a ZeroGPU quota. The weights, 256-token input and exact 28-label map are unchanged. `GRNTI_DEVICE=cuda` explicitly enables the optional GPU wrapper on a ZeroGPU host; HF quotas then apply. No paid hardware is configured.
