"""ICD-10 code validation and prefix-based correction for parse-time use.

Lazy-loads the ICD10Dictionary singleton from etl/ontology/icd10.py.
"""

from dataclasses import dataclass

from etl.utils.logging import get_logger

log = get_logger("eval.icd10_validator")

_icd10_dict = None


def _get_dict():
    global _icd10_dict
    if _icd10_dict is None:
        from etl.ontology.icd10 import ICD10Dictionary
        _icd10_dict = ICD10Dictionary()
    return _icd10_dict


def format_icd10(code: str) -> str:
    """Standard dotted form (I2101 -> I21.01); anything of 3 characters or fewer is unchanged."""
    norm = (code or "").replace(".", "").replace(" ", "").upper()
    return f"{norm[:3]}.{norm[3:]}" if len(norm) > 3 else norm


def is_valid_icd10(code: str) -> bool:
    """Check if normalized code exists in ICD-10-CM 2025."""
    if not code:
        return False
    return _get_dict().lookup_code(code) is not None


def nearest_valid_icd10(code: str) -> str | None:
    """Try prefix-based correction for an invalid code.

    Fallback chain:
      1. Progressive truncation from len-1 down to 4 chars → first billable match
      2. 3-char parent category (may be header)
      3. None if nothing matches
    """
    d = _get_dict()
    norm = code.replace(".", "").replace(" ", "").upper()
    if not norm or len(norm) < 3:
        return None

    # Try progressively shorter truncations (prefer billable)
    for length in range(len(norm) - 1, 3, -1):
        entry = d.lookup_code(norm[:length])
        if entry and not entry.is_header:
            return norm[:length]

    # Fall back to 3-char parent (header)
    entry = d.lookup_code(norm[:3])
    if entry:
        return norm[:3]

    return None


# Lexical name-based resolution. Two routes, in order:
#   1. SNOMED CT: name -> matching concept -> ICD-10-CM via the Extended Map. SNOMED has the
#      synonyms ICD descriptions lack. It also vouches for a submitted code the name's concept maps
#      to (or a parent/child of it), which protects synonym-named correct codes from replacement.
#   2. ICD-10-CM descriptions: lexical match against the dictionary.

RESOLVE_MIN_NAME_SCORE = 0.90      # best candidate vs. the stated name
RESOLVE_MAX_CURRENT_SCORE = 0.10   # submitted code's own description vs. the name
RESOLVE_MIN_MARGIN = 0.10          # best candidate over the runner-up

USE_SNOMED = True
SNOMED_MIN_SCORE = 0.90            # concept match vs. the stated name (exact hits score 1.0)
SNOMED_MIN_MARGIN = 0.10           # best concept over the best one mapping to a different code
SNOMED_CONSISTENT_MIN_SCORE = 0.70 # concepts this close to the name can vouch for a submitted code

_snomed_map = None
_snomed_unavailable = False


@dataclass(frozen=True)
class Resolution:
    code: str                    # undotted ICD-10-CM code to use
    via: str                     # "snomed_exact" | "snomed_fuzzy" | "icd_lexical"
    snomed_id: str | None = None


def _get_snomed():
    """Lazy SNOMED->ICD-10 index; None (route disabled) if the release files are missing."""
    global _snomed_map, _snomed_unavailable
    if _snomed_map is None and not _snomed_unavailable:
        try:
            from etl.ontology.snomed_icd10 import SnomedICD10Map
            _snomed_map = SnomedICD10Map()
        except (OSError, KeyError) as e:
            _snomed_unavailable = True
            log.warning("SNOMED name resolution disabled: %s", e)
    return _snomed_map


def _icd_from_snomed_target(target: str) -> str | None:
    """A SNOMED map target as a code in our ICD-10-CM dictionary.

    Targets can carry a "?" placeholder for an injury 7th character, or come from a newer ICD-10-CM
    release than our dictionary; both are cut back to the nearest code we have.
    """
    code = target.replace("?", "")
    if code and not is_valid_icd10(code):
        code = nearest_valid_icd10(code)
    entry = _get_dict().lookup_code(code) if code else None
    return entry.code if entry else None


def _snomed_vouches(code: str, matches) -> bool:
    """True if a concept close to the name maps to the submitted code, or a parent or child of it.

    A sibling leaf in the same category (J84.0 vs J84.112) does not count: that is the usual
    near-miss we want to correct.
    """
    norm = code.replace(".", "").replace(" ", "").upper()
    if len(norm) < 3:
        return False
    for m in matches:
        if m.score < SNOMED_CONSISTENT_MIN_SCORE:
            continue
        for target in m.all_targets:
            target = target.replace("?", "")
            if target.startswith(norm) or norm.startswith(target):
                return True
    return False


# Injury, poisoning and external-cause targets carry context (encounter, intent, sequela) that a
# diagnosis name alone does not; SNOMED defaults there are not trusted.
_UNTRUSTED_CHAPTERS = frozenset("STVWXY")


def _snomed_candidate(matches) -> Resolution | None:
    """The single ICD-10-CM code the SNOMED matches agree on, or None if unclear."""
    usable = [(m, _icd_from_snomed_target(m.default_target)) for m in matches if m.default_target]
    usable = [(m, c) for m, c in usable if c]
    if not usable:
        return None
    best, best_code = usable[0]
    if best_code[0] in _UNTRUSTED_CHAPTERS:
        return None
    others = [m.score for m, c in usable if c != best_code]
    if best.score < SNOMED_MIN_SCORE or (others and best.score - others[0] < SNOMED_MIN_MARGIN):
        return None
    return Resolution(best_code, f"snomed_{best.via}", best.concept_id)


def resolve_icd10(code: str, name: str) -> Resolution | None:
    """Return a replacement ICD-10-CM code for `name` if `code` should be replaced, else None.

    Uses only the submission, SNOMED CT and the ICD-10-CM dictionary (never the ground truth).
    Works for invalid codes too, in which case the current-code score is 0.
    """
    from eval.semantic_match import _expand_abbreviations, similarity

    name = (name or "").strip()
    if not name:
        return None
    d = _get_dict()

    # check if the ICD code's description matches the proposed description
    entry = d.lookup_code(code) if code else None
    if entry is not None and similarity(name, entry.description) > RESOLVE_MAX_CURRENT_SCORE:
        return None

    # if there wasn't an ICD hit, check SNOMED descriptions and fetch their corresponding ICD code
    snomed = _get_snomed() if USE_SNOMED else None
    if snomed is not None:
        matches = snomed.match(name)
        if code and _snomed_vouches(code, matches):
            return None
        found = _snomed_candidate(matches)
        # A valid code in the same category is at worst a lateral move ("unspecified" variants are
        # defensible), so SNOMED only overrides a code it places in a different category. Same-
        # category near-misses are left to the stricter ICD description match below.
        if found and not (entry is not None and found.code[:3] == entry.code[:3]):
            return found

    # if there wasn't an exact match via SNOMED, fallback to fuzzy matching on ICD terms
    pool = {}
    for query in dict.fromkeys((name, _expand_abbreviations(name))):
        for cand, _ in d.fuzzy_match(query, top_k=10, min_score=0.3):
            pool[cand.code] = cand
    ranked = sorted(((similarity(name, c.description), c.code) for c in pool.values()),
                    key=lambda x: (-x[0], x[1]))
    if not ranked:
        return None
    best_score, best_code = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
    if (best_score < RESOLVE_MIN_NAME_SCORE or best_score - runner_up < RESOLVE_MIN_MARGIN
            or (entry is not None and best_code == entry.code)):
        return None
    return Resolution(best_code, "icd_lexical")


def resolve_icd10_from_name(code: str, name: str) -> str | None:
    """`resolve_icd10`, returning just the replacement code."""
    found = resolve_icd10(code, name)
    return found.code if found else None
