# Dataset Notes

Raw 10-K filings are downloaded from SEC EDGAR by `scripts/build_dataset.py`.

Expected raw layout:

```text
data/raw/10k/
  metadata.csv
  metadata.jsonl
  manifest.json
  <ticker>/
    <filing-date>_<ticker>_<accession>.htm
```

`questions.jsonl` should contain gold QA pairs with category and evidence labels.
