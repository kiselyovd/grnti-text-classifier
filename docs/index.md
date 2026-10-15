# Scientific-Text Classification

Classify Russian scientific abstracts into 28 exact GRNTI codes with XLM-RoBERTa.

**[Classify your own abstract](https://huggingface.co/spaces/kiselyovd/grnti-text-classifier-demo)** · [Browse source subjects and examples](showcase.md) · [Pinned model](https://huggingface.co/kiselyovd/grnti-text-classifier)

## Evaluation

<!-- ARTIFACT-MANIFEST:START -->
### Current artifact and evaluation

**Native checkpoint replay on 2,772 pinned source test abstracts**

| Metric | Value |
|---|---:|
| Top-1 accuracy | 72.37% |
| Top-5 accuracy | 96.75% |
| Macro F1 | 72.30% |

Current pinned source test split. Original training dataset revision and membership were not recorded; held-out independence is not re-established by this replay.

[Evaluation record](https://github.com/kiselyovd/grnti-text-classifier/blob/main/reports/replay-native.json) · [Pinned model](https://huggingface.co/kiselyovd/grnti-text-classifier/tree/24fbe75c27112612a464935c95465def6af788ef)
<!-- ARTIFACT-MANIFEST:END -->

[Replay protocol and component terms](EVALUATION.md). The replay saves source IDs, logits and targets. Historical training membership is unrecorded. The baseline's historical top-1/top-5/F1 remains 72.9%/95.9%/72.8%; its arrays have not been replayed here. Those values alone do not explain why the models differ.

## Numerical evidence

![Per-class F1 from the saved native replay](images/per_class_f1.png)

![Row-normalised confusion matrix from the saved native replay](images/confusion_matrix.png)

Both figures are generated from `reports/replay-native.json` and its hash-bound arrays; code strings preserve the checkpoint's separate agricultural classes.

## Guides

[Architecture](architecture.md) · [Training](training.md) · [Serving](serving.md) · [API](api.md) · [Reproduce this evaluation](EVALUATION.md)
