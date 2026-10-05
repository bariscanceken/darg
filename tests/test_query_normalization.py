"""Guard tests for DARG query typo normalization (stdlib unittest, no network, no OpenAI)."""

from __future__ import annotations

import unittest

from src.rag.query_normalization import (
    METHOD_DIACRITICS,
    METHOD_EDIT_DISTANCE,
    PROTECTED_WORDS,
    QueryVocab,
    changes_only_inflection,
    load_query_vocab,
    normalize_query,
    osa_distance,
    turkish_lower,
)


def normalized(query: str) -> str:
    return normalize_query(query)["normalized"]


class TargetCorrectionTests(unittest.TestCase):
    def test_spec_example_misspelling(self) -> None:
        result = normalize_query("Deprem sırasında asaönser kullanmalı mıyım?")
        self.assertEqual(result["normalized"], "Deprem sırasında asansör kullanmalı mıyım?")
        self.assertEqual([(c["original"], c["corrected"], c["method"]) for c in result["corrections"]],
                         [("asaönser", "asansör", METHOD_EDIT_DISTANCE)])

    def test_spec_example_ascii(self) -> None:
        result = normalize_query("Deprem sırasında asansor kullanmalı mıyım?")
        self.assertEqual(result["normalized"], "Deprem sırasında asansör kullanmalı mıyım?")
        self.assertEqual(result["corrections"][0]["method"], METHOD_DIACRITICS)

    def test_capitalization_is_restored(self) -> None:
        self.assertEqual(normalized("Asansor kullanmalı mıyım?"), "Asansör kullanmalı mıyım?")


class IdentityTests(unittest.TestCase):
    def test_clean_query_returned_byte_for_byte(self) -> None:
        for query in [
            "Deprem sırasında asansör kullanmalı mıyım?",
            "  Deprem   sırasında asansör kullanmalı mıyım ?! ",
            "DEPREM SIRASINDA NE YAPMALIYIM?",
            "",
            "112",
        ]:
            result = normalize_query(query)
            self.assertIs(result["normalized"], query)
            self.assertEqual(result["corrections"], [])

    def test_only_misspelled_span_changes(self) -> None:
        query = "Deprem, sırasında: asaönser (kullanmalı) mıyım?"
        self.assertEqual(normalized(query), "Deprem, sırasında: asansör (kullanmalı) mıyım?")


class FalseCorrectionGuardTests(unittest.TestCase):
    def test_question_particles_are_protected(self) -> None:
        for particle in ["mıyım", "miyim", "musun", "müsünüz", "mıdır", "miyiz"]:
            self.assertIn(particle, PROTECTED_WORDS)
            query = f"Deprem sırasında dışarı çıkmalı {particle}?"
            self.assertEqual(normalized(query), query)

    def test_diacritics_typed_by_user_are_never_removed(self) -> None:
        # Folding "mıyım" would give "miyim"; the user's own "ı" must be kept.
        vocab = QueryVocab({"miyim": 50, "deprem": 100})
        self.assertEqual(normalize_query("deprem mıyım", vocab)["normalized"], "deprem mıyım")

    def test_valid_inflections_missing_from_corpus_are_kept(self) -> None:
        for query in [
            "Deprem sırasında asansör kullanmalıyım mı?",
            "Kalmalıyım mı yoksa çıkmalıyım mı?",
            "Bunu nasıl yapabilirim?",
            "Deprem olduğunda evdeyken ne yapmalıyım?",
            "Asansörle inmek tehlikeli mi?",
            "Evimin depreme dayanıklı olup olmadığını söyleyebilir misin?",
            "Binanın güvenli olup olmadığına kendim karar verebilir miyim?",
        ]:
            self.assertEqual(normalized(query), query, query)

    def test_inflection_only_change_is_rejected(self) -> None:
        self.assertTrue(changes_only_inflection("kendim", "kendi"))
        self.assertTrue(changes_only_inflection("söyleyebilir", "söylenebilir"))
        self.assertFalse(changes_only_inflection("asaönser", "asansör"))

    def test_ambiguous_candidates_are_left_untouched(self) -> None:
        vocab = QueryVocab({"kalemler": 40, "kalemlar": 30, "deprem": 100})
        self.assertEqual(normalize_query("deprem kalemlxr", vocab)["normalized"], "deprem kalemlxr")

    def test_dominant_candidate_is_accepted(self) -> None:
        vocab = QueryVocab({"kalemler": 400, "kalemlar": 30, "deprem": 100})
        self.assertEqual(normalize_query("deprem kalemlxr", vocab)["normalized"], "deprem kalemler")

    def test_short_words_acronyms_and_brands_are_skipped(self) -> None:
        for query in ["AFAD ne yapar?", "WhatsApp kullanmalı mıyım?", "Deprm"]:
            self.assertEqual(normalized(query), query)

    def test_words_adjacent_to_digits_are_skipped(self) -> None:
        query = "7.4büyüklüğündeki deprem"
        self.assertEqual(normalized(query), query)

    def test_mostly_unknown_query_is_left_untouched(self) -> None:
        result = normalize_query("Lorem ipsum dolor asaönser amet")
        self.assertEqual(result["normalized"], "Lorem ipsum dolor asaönser amet")
        self.assertEqual(result["skipped_reason"], "too_many_unknown_words")

    def test_more_than_two_corrections_leaves_query_untouched(self) -> None:
        vocab = QueryVocab({"asansör": 50, "çıkış": 50, "güvenli": 50, "deprem": 100, "sırasında": 100, "ile": 100})
        result = normalize_query("deprem sırasında asansor ile cikis guvenli mi", vocab)
        self.assertEqual(result["normalized"], "deprem sırasında asansor ile cikis guvenli mi")
        self.assertEqual(result["skipped_reason"], "too_many_corrections")


class HelperTests(unittest.TestCase):
    def test_turkish_lower(self) -> None:
        self.assertEqual(turkish_lower("IŞIK İZMİR"), "ışık izmir")

    def test_osa_distance(self) -> None:
        self.assertEqual(osa_distance("asaönser", "asansör", 2), 2)
        self.assertEqual(osa_distance("elektirk", "elektrik", 2), 1)
        self.assertEqual(osa_distance("abcdef", "uvwxyz", 2), 3)

    def test_vocab_artifact_is_versioned(self) -> None:
        self.assertEqual(len(load_query_vocab().source_sha256), 64)


if __name__ == "__main__":
    unittest.main()
