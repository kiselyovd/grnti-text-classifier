# Scientific text, 28 subject codes

**[Classify your own abstract](https://huggingface.co/spaces/kiselyovd/grnti-text-classifier-demo)** · [Replay and numerical evidence](EVALUATION.md)

Browse the model label catalogue and the committed source abstracts. Search a phrase or a subject, then reveal the dataset label to explore how the task is defined.

<section class="sc-explorer" data-showcase="corpus" data-owner="kiselyovd" aria-label="Scientific text, 28 subject codes interactive showcase">
  <div class="sc-controls"></div>
  <div class="sc-workspace">
    <article class="sc-stage sc-record"><h3>Пищевая промышленность</h3><p>Исследование влияния ферментных препаратов на процессы созревания зернового теста и качество хлеба. Досліджено вплив ферментних препаратів целюлази, ксиланази та глюкозооксидази на процеси дозріван- ня зернового тіста та якість зерно- вого хліба. Встановлено, що внесення дослідних ферментних препаратів у зернове тісто під час його приготу- вання сприяє інтенсифікації біохіміч- них і мікробіологічних процесів дозрі- вання. У результаті покращуються реологічні властивості тіста та під- вищуються показники якості гото- вих виробівКлючові слова: ферментні препа- рати, зерновий хліб, приготування тіста, целюлаза, ксиланаза, глюко- зооксидаза</p><p>Source record 10151. Corpus label: 650000. Original text is preserved.</p></article>
    <aside class="sc-details"><h3>Explore the example</h3><p>Use the controls to inspect this project's published artifacts.</p></aside>
  </div>
  <p class="sc-live" role="status">The source example is available above; interactive controls load when JavaScript is enabled.</p>
  
</section>

## Model comparison

| Evaluation evidence | Top-1 accuracy | Top-5 accuracy |
|---|---|---|
| XLM-RoBERTa · native replay | 72.37% | 96.75% |
| ruBERT · historical report | 72.9% | 95.9% |

These operating points answer different questions: one automatic assignment or a candidate list for human review. [Published model card](https://huggingface.co/kiselyovd/grnti-text-classifier/blob/24fbe75c27112612a464935c95465def6af788ef/README.md).

The 20 abstracts carry existing corpus labels. Their original language, code strings, and record IDs are preserved; the model catalogue contains eight additional subjects without a committed sample. The two agriculture-related codes remain separate even though their displayed names match.

## Error-analysis figures

![Published per-class F1 figure](images/per_class_f1.png)

The main-model chart is regenerated from the hash-bound logits and targets in `reports/replay-native.npz`. [Replay protocol](EVALUATION.md). The historical ruBERT prediction arrays remain unavailable.

## Sources and reproduction

The explorer uses published artifacts from [kiselyovd/grnti-text-classifier](https://github.com/kiselyovd/grnti-text-classifier). [Inspect the pinned source](https://github.com/kiselyovd/grnti-text-classifier/blob/35acaa1d9e75e149d97b971137d889a4825206bc/data/sample/sample.jsonl) and the [structured showcase data](assets/showcase/data.json).


## See also

- [Project overview](index.md)
- [Architecture](architecture.md)
- [Source repository](https://github.com/kiselyovd/grnti-text-classifier)
