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

Real XLM-RoBERTa inference on Russian scientific text: five exact codes, full subject names and uncalibrated model scores. English/Russian interface. Runs on free ZeroGPU; availability follows the visitor's HF quota.

Weights: `kiselyovd/grnti-text-classifier@24fbe75c27112612a464935c95465def6af788ef`, SHA256 `324f0ca0b15fd30f8b0c9bd2e9db4594b88119489f3a8d1df0169a455ca1eba4`. Safetensors only; no remote model code. Input: at most 20,000 characters; first 256 tokens; 28 checkpoint labels. Five- and six-digit codes are preserved exactly; 680000 and 683500 intentionally retain their duplicate stored names.

Illustrative examples are author-written, not benchmark ground truth. Scores are not calibrated confidence estimates. No user text is logged or cached by this application. Processing takes place on this public HF Space; avoid confidential inputs.

[Model card](https://huggingface.co/kiselyovd/grnti-text-classifier) · [Source, evaluation and limitations](https://github.com/kiselyovd/grnti-text-classifier)

Local production entry: `python app.py` after installing `requirements.txt`. CPU is used outside HF; model files reuse the normal HF cache. `GRNTI_DEVICE=cpu` explicitly selects CPU. Code MIT; checkpoint's declared license MIT, XLM-R upstream MIT; all upstream package licenses retain their terms.
