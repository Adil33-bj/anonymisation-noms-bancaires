from src.anonymization.mask import mask_spans
from src.detection.dictionary_detector import DictionaryDetector


def test_fixed_phrase_regression() -> None:
    # Regression test: known phrase should remain detectable and maskable.
    detector = DictionaryDetector(
        dict_prenom_to_id={"Yassine": 1},
        dict_nom_to_id={"Kharrouchi": 1},
        use_fuzzy=True,
        fuzzy_threshold=80,
    )
    text = "Mr. Yassine Kharrouchi a payé la facture."
    spans = [{"start": s.start, "end": s.end, "label": s.label} for s in detector.detect(text)]
    out = mask_spans(text, spans)
    assert out == "Mr. [ANONYMIZED] a payé la facture."


def test_repeated_entity_handled() -> None:
    # Repeated entities should produce repeated masks.
    detector = DictionaryDetector(
        dict_prenom_to_id={"Amal": 1},
        dict_nom_to_id={"Bennani": 1},
        use_fuzzy=False,
    )
    text = "Amal Bennani a validé. Amal Bennani a signé."
    spans = [{"start": s.start, "end": s.end, "label": s.label} for s in detector.detect(text)]
    out = mask_spans(text, spans)
    assert out.count("[ANONYMIZED]") == 2
