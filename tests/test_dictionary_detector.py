from src.detection.dictionary_detector import DictionaryDetector, tokenize_with_offsets


def test_tokenize_with_offsets() -> None:
    # Offsets should map correctly to original text slices.
    text = "Mr. Yassine Kharrouchi"
    tokens = tokenize_with_offsets(text)
    assert tokens[0][0] == "Mr"
    assert text[tokens[1][1] : tokens[1][2]] == "Yassine"
    assert text[tokens[2][1] : tokens[2][2]] == "Kharrouchi"


def test_exact_detection_pair() -> None:
    # Exact prenom+nom should produce one PERSON span.
    detector = DictionaryDetector(
        dict_prenom_to_id={"Ahmed": 1},
        dict_nom_to_id={"Benali": 1},
        use_fuzzy=False,
    )
    spans = detector.detect("Virement de Ahmed Benali vers compte.")
    assert len(spans) == 1
    assert spans[0].text.lower() == "ahmed benali"


def test_fuzzy_detection_single() -> None:
    # Typo should still be recognized in fuzzy mode.
    detector = DictionaryDetector(
        dict_prenom_to_id={"Ahmed": 1},
        dict_nom_to_id={"Benali": 1},
        use_fuzzy=True,
        fuzzy_threshold=80,
    )
    spans = detector.detect("Paiement pour Ahmedi")
    assert len(spans) >= 1


def test_stopword_not_detected() -> None:
    # Stopwords should not be falsely predicted as PERSON.
    detector = DictionaryDetector(
        dict_prenom_to_id={"Ahmed": 1},
        dict_nom_to_id={"Benali": 1},
        use_fuzzy=False,
    )
    spans = detector.detect("Le et de la du un une")
    assert spans == []


def test_particle_el_kept_in_name() -> None:
    # « Yassine El Mansouri » doit être détecté en un seul nom (avec « El »).
    detector = DictionaryDetector(
        dict_prenom_to_id={"Yassine": 1},
        dict_nom_to_id={"Mansouri": 1},
        use_fuzzy=False,
    )
    spans = detector.detect("Virement pour Yassine El Mansouri.")
    assert len(spans) == 1
    assert spans[0].text == "Yassine El Mansouri"
