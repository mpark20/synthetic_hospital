"""The env submit path and the eval harness share one problem-list normalization."""

import json

from epic_sim.app.services.env_service import normalize_submission
from eval.icd10_validator import format_icd10, is_valid_icd10
from eval.tasks.patient_diagnosis import normalize_problem_list, parse_output

BAD_CODE = "E11.999"  # real category, nonexistent leaf


def _entry(code, name="x", acuity="acute"):
    return {"icd10": code, "name": name, "acuity": acuity}


def test_bad_code_is_not_valid_fixture():
    assert not is_valid_icd10(BAD_CODE)


def test_normalize_problem_list_matches_parse_output():
    data = {
        "active_diagnoses": [_entry(BAD_CODE), _entry("I21.4", acuity="Acute")],
        "chronic_conditions": [_entry("I21.4", acuity="chronic"), _entry("N18.3", acuity="chronic")],
    }
    parsed = parse_output(json.dumps(data))
    active, chronic = normalize_problem_list(data["active_diagnoses"], data["chronic_conditions"])
    undot = lambda es: [{**e, **{k: e[k].replace(".", "") for k in ("icd10", "icd10_original") if k in e}} for e in es]
    assert parsed["active_diagnoses"] == undot(active)
    assert parsed["chronic_conditions"] == undot(chronic)


def test_codes_leave_in_dotted_form():
    active, _ = normalize_problem_list([_entry("I214"), _entry("I10"), _entry(BAD_CODE)], [])
    assert [e["icd10"] for e in active][:2] == ["I21.4", "I10"]
    assert active[2]["icd10_original"] == "E11.999"


def test_env_submission_corrects_invalid_code_and_keeps_original():
    pred = normalize_submission("patient_diagnosis", {
        "active_diagnoses": [_entry(BAD_CODE, "type 2 diabetes")], "chronic_conditions": []})
    dx = pred["active_diagnoses"][0]
    assert dx["icd10_corrected"] and dx["icd10_original"] == "E11.999"
    assert is_valid_icd10(dx["icd10"]) and dx["icd10"].startswith("E11")


def test_env_submission_leaves_valid_code_untouched():
    pred = normalize_submission("patient_diagnosis", {
        "active_diagnoses": [_entry("I21.4")], "chronic_conditions": []})
    dx = pred["active_diagnoses"][0]
    assert dx["icd10"] == "I21.4" and "icd10_corrected" not in dx


def test_env_submission_dedups_across_lists_like_harness():
    pred = normalize_submission("patient_diagnosis", {
        "active_diagnoses": [_entry("I21.4")],
        "chronic_conditions": [_entry("I21.4", acuity="chronic")]})
    assert len(pred["active_diagnoses"]) == 1 and pred["chronic_conditions"] == []


def test_env_submission_handles_wrapper_bare_list_and_lone_entry():
    for obj in ({"payload": {"active_diagnoses": [_entry(BAD_CODE)]}},
                [_entry(BAD_CODE)], _entry(BAD_CODE)):
        pred = normalize_submission("patient_diagnosis", obj)
        assert pred["active_diagnoses"][0]["icd10_corrected"]
        assert pred["chronic_conditions"] == []


def test_env_submission_tolerates_null_lists_and_malformed():
    pred = normalize_submission("patient_diagnosis", {"active_diagnoses": None, "chronic_conditions": []})
    assert pred == {"active_diagnoses": [], "chronic_conditions": []}
    assert normalize_submission("patient_diagnosis", {"nonsense": 1}) is None
    assert normalize_submission("patient_diagnosis", {}) is None


def test_other_tasks_are_unchanged():
    assert normalize_submission("context_summarization", {"summary": "s"}) == {"summary": "s"}
    assert normalize_submission("evidence_retrieval", [{"a": 1}]) == {"rankings": [{"a": 1}]}


# --- name-based code resolution (env only) ---------------------------------

from eval.icd10_validator import nearest_valid_icd10, resolve_icd10_from_name  # noqa: E402


def test_resolves_hallucinated_valid_codes_from_name():
    assert resolve_icd10_from_name("J84.0", "Idiopathic pulmonary fibrosis") == "J84.112"
    assert resolve_icd10_from_name("G50.0", "Bell's palsy") == "G51.0"
    assert resolve_icd10_from_name("K90.2", "Celiac disease") == "K90.0"
    assert resolve_icd10_from_name("E87.0", "Hyperkalemia") == "E87.5"


def test_never_overrides_codes_that_were_right_but_lexically_distant():
    # Correct (or defensible) codes from frontier-model traces whose names do not
    # lexically match the official description; the resolver must leave them.
    for code, name in [("F32.A", "Major depressive disorder"),
                       ("F10.21", "Alcohol use disorder, in remission"),
                       ("K27.9", "Peptic ulcer disease"),
                       ("E83.39", "Hypophosphatemia"),
                       ("E05.00", "Graves' disease"),
                       ("M85.80", "Osteopenia"),
                       ("B35.1", "Onychomycosis"),
                       ("I10", "Essential hypertension"),
                       ("Z87.891", "Former smoker")]:
        assert resolve_icd10_from_name(code, name) is None, (code, name)


def test_ambiguous_name_is_not_resolved_and_falls_back_to_prefix():
    # "Type 2 diabetes mellitus" fits many E11.x codes equally: no single winner.
    assert resolve_icd10_from_name(BAD_CODE, "Type 2 diabetes mellitus") is None
    pred = normalize_submission("patient_diagnosis", {
        "active_diagnoses": [_entry(BAD_CODE, "Type 2 diabetes mellitus")], "chronic_conditions": []})
    dx = pred["active_diagnoses"][0]
    assert dx["icd10_corrected"] and "icd10_resolved" not in dx
    assert dx["icd10"] == format_icd10(nearest_valid_icd10(BAD_CODE))


def test_empty_name_or_code_does_not_resolve_or_crash():
    assert resolve_icd10_from_name("J84.0", "") is None
    assert resolve_icd10_from_name("", "") is None
    assert resolve_icd10_from_name("", "Bell's palsy") == "G51.0"


def test_resolution_is_deterministic():
    results = {resolve_icd10_from_name("J84.0", "Idiopathic pulmonary fibrosis") for _ in range(5)}
    assert results == {"J84.112"}


def test_env_submission_resolves_and_records_original():
    pred = normalize_submission("patient_diagnosis", {
        "active_diagnoses": [_entry("G50.0", "Bell's palsy")], "chronic_conditions": []})
    dx = pred["active_diagnoses"][0]
    assert dx["icd10"] == "G51.0" and dx["icd10_original"] == "G50.0" and dx["icd10_resolved"]


def test_harness_default_does_not_resolve():
    active, _ = normalize_problem_list([_entry("G50.0", "Bell's palsy")], [])
    assert active[0]["icd10"] == "G50.0" and "icd10_resolved" not in active[0]


# --- SNOMED CT route ---------------------------------------------------------

import pytest  # noqa: E402

from eval import icd10_validator  # noqa: E402
from eval.icd10_validator import resolve_icd10  # noqa: E402
from etl.ontology.snomed_icd10 import SnomedICD10Map  # noqa: E402


@pytest.fixture(scope="module")
def snomed():
    return SnomedICD10Map()


def test_snomed_synonym_resolves_code_the_icd_matcher_cannot():
    r = resolve_icd10("", "Graves' disease")
    assert r and r.via == "snomed_exact" and r.code == "E05.00" and r.snomed_id


def test_snomed_replaces_code_from_a_different_category():
    r = resolve_icd10("J18.9", "Hyperkalemia")
    assert r and r.via.startswith("snomed") and r.code == "E87.5"


def test_snomed_does_not_override_a_same_category_code():
    assert resolve_icd10("F32.A", "Major depressive disorder") is None


def test_snomed_vouches_for_exact_and_parent_codes():
    assert resolve_icd10("E05.00", "Graves' disease") is None
    assert resolve_icd10("E05", "Graves' disease") is None


def test_snomed_route_can_be_disabled(monkeypatch):
    monkeypatch.setattr(icd10_validator, "USE_SNOMED", False)
    assert resolve_icd10("", "Graves' disease") is None


def test_snomed_route_degrades_when_release_is_missing(monkeypatch):
    monkeypatch.setattr(icd10_validator, "_snomed_map", None)
    monkeypatch.setattr(icd10_validator, "_snomed_unavailable", True)
    assert resolve_icd10("", "Graves' disease") is None
    assert resolve_icd10("G50.0", "Bell's palsy").code == "G51.0"   # ICD route still works


def test_snomed_target_placeholders_and_unknown_codes_fall_back_to_a_known_parent():
    code = icd10_validator._icd_from_snomed_target("S68612?")
    assert code and icd10_validator.is_valid_icd10(code) and code.replace(".", "").startswith("S686")
    assert icd10_validator._icd_from_snomed_target("") is None


def test_context_dependent_concepts_use_the_default_branch_and_unclassifiable_have_none(snomed):
    hk = snomed.match("Hyperkalemia", top_k=1)[0]
    assert hk.default_target == "E875" and len(hk.all_targets) > 1
    assert all(m.default_target for m in snomed.match("Bell's palsy"))


def test_snomed_matching_is_exact_then_abbreviation_aware_then_fuzzy(snomed):
    assert snomed.match("graves disease")[0].via == "exact"
    assert snomed.match("HTN")[0].term == "hypertension"
    assert snomed.match("stemi of anterior wall")[0].score >= 0.9
    assert snomed.match("") == [] and snomed.match("zzzz qqqq") == []


def test_env_submission_records_snomed_provenance():
    pred = normalize_submission("patient_diagnosis", {
        "active_diagnoses": [_entry("J18.9", "Hyperkalemia")], "chronic_conditions": []})
    dx = pred["active_diagnoses"][0]
    assert dx["icd10"] == "E87.5" and dx["icd10_resolved_via"].startswith("snomed") and dx["snomed_id"]


def test_history_names_do_not_fuzzy_match_the_active_disease(snomed):
    assert all("history" in m.term.lower() for m in snomed.match("History of clear cell renal cell carcinoma"))
    assert resolve_icd10("Z85.528", "History of clear cell renal cell carcinoma") is None


def test_injury_chapter_snomed_defaults_are_not_used():
    assert resolve_icd10("E87.8", "Refeeding syndrome") is None
