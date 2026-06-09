from videorag._entity_anchor import text_entities


def _words(text, start=0.0):
    words = []
    cursor = float(start)
    for token in text.split():
        words.append(
            {
                "text": token,
                "start": cursor,
                "end": cursor + 0.2,
            }
        )
        cursor += 0.25
    return words


def test_transcript_entities_do_not_create_speakers_or_claims(monkeypatch):
    monkeypatch.setattr(
        text_entities,
        "_load_nlp",
        lambda config: (None, "regex_fallback"),
    )
    extractor = text_entities.TextEntityExtractor({})

    result = extractor.extract_segment(
        {"segment_id": "SEG_00000"},
        [
            {"text": "John", "start": 0.0, "end": 0.4},
            {"text": "holds", "start": 0.5, "end": 0.8},
            {"text": "the", "start": 0.9, "end": 1.0},
            {"text": "skirt", "start": 1.1, "end": 1.4},
        ],
    )

    assert "claims" not in result
    assert "speaker" not in result
    assert "SPEAKER" not in result["rewritten_transcript"]
    assert result["mentions"][0]["text_id"] == "T_PERSON_001"


def test_pronoun_resolves_from_previous_segment_memory(monkeypatch):
    monkeypatch.setattr(
        text_entities,
        "_load_nlp",
        lambda config: (None, "regex_fallback"),
    )
    extractor = text_entities.TextEntityExtractor({})
    first = extractor.extract_segment(
        {
            "segment_id": "SEG_00000",
            "index": 0,
            "start": 0.0,
            "end": 2.0,
        },
        _words("A man arrived."),
    )
    second = extractor.extract_segment(
        {
            "segment_id": "SEG_00001",
            "index": 1,
            "start": 3.0,
            "end": 5.0,
        },
        _words("He sat down.", start=3.0),
    )

    entity_id = first["mentions"][0]["text_id"]
    assert second["references"][0]["status"] == "resolved"
    assert second["references"][0]["resolved_text_id"] == entity_id
    assert f"[{entity_id}] sat down." in second["rewritten_transcript"]
    assert second["mentions"][0]["reference_forms"] == ["He"]


def test_ambiguous_pronoun_is_preserved_for_alignment(monkeypatch):
    monkeypatch.setattr(
        text_entities,
        "_load_nlp",
        lambda config: (None, "regex_fallback"),
    )
    extractor = text_entities.TextEntityExtractor({})
    extractor.extract_segment(
        {
            "segment_id": "SEG_00000",
            "index": 0,
            "start": 0.0,
            "end": 2.0,
        },
        _words("John met David."),
    )
    result = extractor.extract_segment(
        {
            "segment_id": "SEG_00001",
            "index": 1,
            "start": 3.0,
            "end": 5.0,
        },
        _words("He left.", start=3.0),
    )

    reference = result["references"][0]
    assert reference["status"] == "unresolved"
    assert len(reference["candidates"]) == 2
    assert "<unresolved-ref" in result["rewritten_transcript"]
    assert result["mentions"] == []


def test_text_memory_survives_serialization(monkeypatch):
    monkeypatch.setattr(
        text_entities,
        "_load_nlp",
        lambda config: (None, "regex_fallback"),
    )
    extractor = text_entities.TextEntityExtractor({})
    first = extractor.extract_segment(
        {
            "segment_id": "SEG_00000",
            "index": 0,
            "start": 0.0,
            "end": 2.0,
        },
        _words("A woman arrived."),
    )
    resumed = text_entities.TextEntityExtractor(
        {},
        state=extractor.to_state(),
    )
    second = resumed.extract_segment(
        {
            "segment_id": "SEG_00001",
            "index": 1,
            "start": 3.0,
            "end": 5.0,
        },
        _words("She smiled.", start=3.0),
    )

    assert (
        second["references"][0]["resolved_text_id"]
        == first["mentions"][0]["text_id"]
    )


def test_no_transcript_memory_keeps_exact_names_but_not_pronouns(monkeypatch):
    monkeypatch.setattr(
        text_entities,
        "_load_nlp",
        lambda config: (None, "regex_fallback"),
    )
    extractor = text_entities.TextEntityExtractor(
        {"disable_transcript_memory": True}
    )
    first = extractor.extract_segment(
        {
            "segment_id": "SEG_00000",
            "index": 0,
            "start": 0.0,
            "end": 2.0,
        },
        _words("John arrived."),
    )
    second = extractor.extract_segment(
        {
            "segment_id": "SEG_00001",
            "index": 1,
            "start": 3.0,
            "end": 5.0,
        },
        _words("John said he left.", start=3.0),
    )

    assert first["mentions"][0]["text_id"] == second["mentions"][0]["text_id"]
    assert second["references"][0]["status"] == "unresolved"
    assert second["references"][0]["candidates"] == []
    assert second["memory_context"]["long_term"]["entities"] == []
