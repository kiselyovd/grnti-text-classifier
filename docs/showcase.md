# Scientific text, 28 subject codes

Browse the model label catalogue and the committed source abstracts. Search a phrase or a subject, then reveal the dataset label to explore how the task is defined.

<section class="sc-explorer" data-showcase="corpus" data-owner="kiselyovd" aria-label="Scientific text, 28 subject codes interactive showcase">
  <div class="sc-controls"></div>
  <div class="sc-workspace">
    <article class="sc-stage sc-record"><h3>Labelled scientific abstracts</h3><p>The source corpus contains 20 committed examples. Its original text and labels are preserved in the structured data linked below.</p></article>
    <aside class="sc-details"><h3>Explore the example</h3><p>Use the controls to inspect this project's published artifacts.</p></aside>
  </div>
  <p class="sc-live" role="status">The source example is available above; interactive controls load when JavaScript is enabled.</p>
  
</section>

## Model comparison

| Published model-card results | Top-1 accuracy | Top-5 accuracy |
|---|---|---|
| XLM-RoBERTa | 72.4% | 96.8% |
| ruBERT | 72.9% | 95.9% |

These operating points answer different questions: one automatic assignment or a candidate list for human review. [Published model card](https://huggingface.co/kiselyovd/grnti-text-classifier/blob/24fbe75c27112612a464935c95465def6af788ef/README.md).

The 20 abstracts carry existing corpus labels. Their original language, code strings, and record IDs are preserved; the model catalogue contains eight additional subjects without a committed sample. The two agriculture-related codes remain separate even though their displayed names match.

## Error-analysis figures

![Published per-class F1 figure](images/per_class_f1.png)

The published chart is preserved as an image. Its underlying numerical array is not present in the tracked repository, so the explorer does not derive new per-class scores from it.

## Sources and reproduction

The explorer uses published artifacts from [kiselyovd/grnti-text-classifier](https://github.com/kiselyovd/grnti-text-classifier). [Inspect the pinned source](https://github.com/kiselyovd/grnti-text-classifier/blob/35acaa1d9e75e149d97b971137d889a4825206bc/data/sample/sample.jsonl) and the [structured showcase data](assets/showcase/data.json).

UI state stays in the current page. Detailed diagnostic events can be enabled with `?debug=1`; they contain selected IDs and modes, not sample content.

## See also

- [Project overview](index.md)
- [Architecture](architecture.md)
- [Source repository](https://github.com/kiselyovd/grnti-text-classifier)
