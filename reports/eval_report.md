# Evaluation Report

## Model comparison
| Model | Precision | Recall | F1 | False Positive Rate |
|---|---:|---:|---:|---:|
| Dictionary exact | 0.8132 | 0.8132 | 0.8132 | 0.1868 |
| Dictionary fuzzy (t=80) | 0.7946 | 0.9780 | 0.8768 | 0.2054 |
| spaCy NER | 0.9783 | 0.9890 | 0.9836 | 0.0217 |

## Dev threshold sweep (dictionary fuzzy)
- t=75: P=0.3933, R=0.7195, F1=0.5086, FPR=0.6067
- t=80: P=0.7248, R=0.9634, F1=0.8272, FPR=0.2752
- t=85: P=0.7529, R=0.7805, F1=0.7665, FPR=0.2471
- t=90: P=0.7619, R=0.7805, F1=0.7711, FPR=0.2381

## Anonymization metrics
- Split based (dictionary fuzzy): masking_recall=1.0000, overmask_rate=0.1875
- Split based (spaCy): masking_recall=0.9890, overmask_rate=0.0109
- Paired gold (dictionary fuzzy): masking_recall=1.0000, overmask_rate=0.1875
- Paired gold (spaCy): masking_recall=0.9890, overmask_rate=0.0109

## Error analysis
### Frequent false positives by category
- raw_json:ner__test_corpus__synthetic_text_with_names|easy|na: 7
- raw_json:ner__test_corpus__synthetic_text_with_names|hard|na: 4
- raw_json:ner_train_data|easy|na: 4
- raw_json:ner__test_corpus__synthetic_text_with_names|medium|na: 3
- raw_txt:ner__input|medium|na: 2

### Frequent false negatives by category
- raw_txt:ner__input|medium|na: 1
- raw_json:ner_train_data|medium|na: 1

### Example false positives
- [raw_txt:ner__input_00004] fp=2 :: Virement de 1500 MAD effectué pour Jamila Oumhani le 12/04/2024.
- [raw_txt:ner__input_00021] fp=1 :: Yassine El Mansouri souhaite faire opposition à un paiement.
- [raw_json:ner__test_corpus__synthetic_text_with_names_00203] fp=1 :: Faire un virement à Amal Bennani
- [raw_json:ner__test_corpus__synthetic_text_with_names_00230] fp=1 :: Faire un virement à Amal Kharrouchi
- [raw_json:ner__test_corpus__synthetic_text_with_names_00243] fp=1 :: Faire un virement à Omar Kharrouchi

### Example false negatives
- [raw_txt:ner__input_00004] fn=1 :: Virement de 1500 MAD effectué pour Jamila Oumhani le 12/04/2024.
- [raw_json:ner_train_data_01226] fn=1 :: Paiement destiné à Jamila Oumhani.