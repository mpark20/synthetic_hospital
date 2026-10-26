"""Lexical SNOMED CT name lookup that resolves to ICD-10-CM codes.

Given a free-text diagnosis name, finds matching SNOMED CT concepts (exact synonym hit, then
abbreviation-aware lexical similarity over every active synonym) and maps each concept to ICD-10-CM
through the US Extended Map refset. SNOMED carries far more synonyms than ICD-10-CM descriptions
("Graves' disease", "Bell's palsy"), so it recovers codes the ICD-only lexical matcher scores ~0 on.

Only concepts that have an ICD-10-CM map target are indexed. Two target sets are kept per concept:
  default_target  the code the map gives when none of its patient-context rules (age, sex, cause...)
                  apply: group 1's last-priority row, rule TRUE / OTHERWISE TRUE. None when that row
                  has no target (the concept cannot be classified)
  all_targets     every code any active map row points at; used to ask "is this code consistent
                  with the name?", never to pick a replacement

No embeddings; stdlib plus eval.semantic_match for scoring. The parsed index is cached as a pickle
next to the release (invalidated when a source file changes).
"""

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from etl.ontology.snomed import (
    DESCRIPTION_FILE, EXTENDED_MAP_FILE, CONCEPT_FILE, FSN_TYPE, SNOMED_DIR, SYNONYM_TYPE,
)
from etl.utils.logging import get_logger

log = get_logger("etl.ontology.snomed_icd10")

CACHE_FILE = SNOMED_DIR.parent.parent / "snomed_icd10_index.pkl"
CACHE_VERSION = 2

# FSN semantic tags whose concepts can name a diagnosis.
SEARCHABLE_TAGS = frozenset({"disorder", "finding", "morphologic abnormality", "event", "situation"})
# The Extended Map file also carries the WHO ICD-10 map (447562003); only ICD-10-CM rows are wanted.
ICD10CM_REFSET = "6011000124106"
NO_TARGET_CATEGORY = "447638001"   # MAP SOURCE CONCEPT CANNOT BE CLASSIFIED WITH AVAILABLE DATA
DEFAULT_RULES = frozenset({"TRUE", "OTHERWISE TRUE"})

_TAG_RE = re.compile(r"\s*\(([^)]+)\)\s*$")
_PLAIN_RE = re.compile(r"[a-z0-9]+")
MAX_CANDIDATES = 300
# Words the similarity scorer treats as stopwords but that change what the diagnosis is: a fuzzy
# match must carry the same ones ("History of renal cell carcinoma" is not "Renal cell carcinoma").
QUALIFIERS = frozenset({"history", "status", "post", "family", "screening", "risk", "suspected",
                        "possible", "probable", "resolved", "no", "without", "negative", "absent"})


@dataclass(frozen=True)
class SnomedMatch:
    concept_id: str
    term: str                    # the matching description
    score: float                 # 1.0 for an exact hit, else lexical similarity
    via: str                     # "exact" | "fuzzy"
    default_target: str | None   # undotted ICD-10-CM code, may carry "?" placeholders
    all_targets: frozenset       # undotted codes, any map row


def _plain(text: str) -> str:
    return " ".join(_PLAIN_RE.findall(text.lower()))


def _clean_target(code: str) -> str:
    return code.replace(".", "").replace(" ", "").upper()


def _rows(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            yield dict(zip(header, line.rstrip("\n").split("\t")))


def _source_stamp() -> list:
    return [(p.name, p.stat().st_size, int(p.stat().st_mtime))
            for p in (CONCEPT_FILE, DESCRIPTION_FILE, EXTENDED_MAP_FILE)]


class SnomedICD10Map:
    """In-memory index: name -> SNOMED concept -> ICD-10-CM code."""

    def __init__(self, use_cache: bool = True):
        self.terms: dict[str, list[str]] = {}          # concept_id -> distinct descriptions
        self.default_target: dict[str, str] = {}       # concept_id -> undotted ICD code
        self.all_targets: dict[str, frozenset] = {}    # concept_id -> undotted ICD codes
        self._exact: dict[str, list[str]] = {}         # plain phrase -> [concept_ids]
        self._tokens: dict[str, set[str]] = {}         # content token -> {concept_ids}
        self._min_len: dict[str, int] = {}             # concept_id -> fewest tokens in any term

        if use_cache and self._load_cache():
            return
        self._build()
        if use_cache:
            self._save_cache()

    # --- build ---------------------------------------------------------

    def _build(self) -> None:
        active = {r["id"] for r in _rows(CONCEPT_FILE) if r["active"] == "1"}
        defaults, targets = self._read_map(active)
        self.default_target, self.all_targets = defaults, targets

        tags: dict[str, str] = {}
        raw_terms: dict[str, set[str]] = {}
        for r in _rows(DESCRIPTION_FILE):
            cid = r["conceptId"]
            if r["active"] != "1" or cid not in targets or r["languageCode"] != "en":
                continue
            term = r["term"]
            if r["typeId"] == FSN_TYPE:
                m = _TAG_RE.search(term)
                tags[cid] = m.group(1) if m else ""
                term = _TAG_RE.sub("", term)
            elif r["typeId"] != SYNONYM_TYPE:
                continue
            raw_terms.setdefault(cid, set()).add(term.strip())

        for cid, terms in raw_terms.items():
            if tags.get(cid) not in SEARCHABLE_TAGS:
                continue
            self.terms[cid] = sorted(terms)
        self.default_target = {c: t for c, t in self.default_target.items() if c in self.terms}
        self.all_targets = {c: t for c, t in self.all_targets.items() if c in self.terms}
        self._index()
        log.info("SNOMED->ICD-10 index: %d concepts (%d with an unambiguous default code), %d phrases",
                 len(self.terms), len(self.default_target), len(self._exact))

    @staticmethod
    def _read_map(active: set) -> tuple[dict, dict]:
        rows: dict[str, list[dict]] = {}
        for r in _rows(EXTENDED_MAP_FILE):
            if (r["active"] == "1" and r["refsetId"] == ICD10CM_REFSET
                    and r["referencedComponentId"] in active):
                rows.setdefault(r["referencedComponentId"], []).append(r)
        defaults, all_targets = {}, {}
        for cid, rs in rows.items():
            codes = {_clean_target(r["mapTarget"]) for r in rs
                     if r["mapTarget"].strip() and r["mapCategoryId"] != NO_TARGET_CATEGORY}
            if not codes:
                continue
            all_targets[cid] = frozenset(codes)
            group1 = [r for r in rs if r["mapGroup"] == "1"]
            last = max(group1, key=lambda r: int(r["mapPriority"]), default=None)
            if (last and last["mapRule"].strip() in DEFAULT_RULES and last["mapTarget"].strip()
                    and last["mapCategoryId"] != NO_TARGET_CATEGORY):
                defaults[cid] = _clean_target(last["mapTarget"])
        return defaults, all_targets

    def _index(self) -> None:
        from eval.semantic_match import _STOPWORDS
        self._exact, self._tokens, self._min_len = {}, {}, {}
        for cid, terms in self.terms.items():
            shortest = 99
            for term in terms:
                phrase = _plain(term)
                if not phrase:
                    continue
                self._exact.setdefault(phrase, []).append(cid)
                toks = {t for t in phrase.split() if len(t) > 2 and t not in _STOPWORDS}
                shortest = min(shortest, len(toks))
                for t in toks:
                    self._tokens.setdefault(t, set()).add(cid)
            self._min_len[cid] = shortest

    # --- cache ---------------------------------------------------------

    def _load_cache(self) -> bool:
        try:
            with open(CACHE_FILE, "rb") as f:
                blob = pickle.load(f)
            if blob["version"] != CACHE_VERSION or blob["stamp"] != _source_stamp():
                return False
            self.terms, self.default_target, self.all_targets = blob["terms"], blob["default"], blob["all"]
        except Exception:
            return False
        self._index()
        return True

    def _save_cache(self) -> None:
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump({"version": CACHE_VERSION, "stamp": _source_stamp(), "terms": self.terms,
                             "default": self.default_target, "all": self.all_targets}, f)
        except OSError as e:
            log.warning("Could not write SNOMED index cache: %s", e)

    # --- query ---------------------------------------------------------

    def _match(self, cid: str, term: str, score: float, via: str) -> SnomedMatch:
        return SnomedMatch(cid, term, score, via, self.default_target.get(cid), self.all_targets[cid])

    def match(self, name: str, top_k: int = 5, min_score: float = 0.5) -> list[SnomedMatch]:
        """Concepts whose descriptions match `name`, best first (exact hits score 1.0)."""
        from eval.semantic_match import _STOPWORDS, _expand_abbreviations, _normalized_phrase, similarity

        name = (name or "").strip()
        if not name:
            return []

        # 1. Exact pass: look up the plain and abbreviation-normalized forms of the name in the
        #    phrase index (deduped, order kept). Any hit short-circuits the fuzzy search.
        exact = {}
        for phrase in dict.fromkeys((_plain(name), _normalized_phrase(name))):
            for cid in self._exact.get(phrase, ()):
                exact.setdefault(cid, phrase)
        if exact:
            return [self._match(cid, p, 1.0, "exact") for cid, p in sorted(exact.items())][:top_k]

        # 2. Fuzzy pass, candidate generation: tokenize the query (abbreviations expanded, short
        #    words and stopwords dropped) and count, per concept, how many query tokens it shares.
        q_tokens = {t for t in _plain(_expand_abbreviations(name)).split() if len(t) > 2 and t not in _STOPWORDS}
        if not q_tokens:
            return []
        overlap: dict[str, int] = {}
        for t in q_tokens:
            for cid in self._tokens.get(t, ()):
                overlap[cid] = overlap.get(cid, 0) + 1
        # Require 2 shared tokens (1 if the query only has one); rank by overlap, then concepts
        # with shorter terms, then id for determinism; cap the list to bound scoring cost.
        need = min(2, len(q_tokens))
        candidates = sorted(
            (c for c, n in overlap.items() if n >= need),
            key=lambda c: (-overlap[c], self._min_len[c], c)
        )[:MAX_CANDIDATES]

        # 3. Fuzzy pass, scoring: only compare against terms with exactly the same qualifier words
        #    as the query ("history of X" must not match "X"), then keep each concept's best term.
        wanted = QUALIFIERS & set(_plain(name).split())
        scored = []
        for cid in candidates:
            terms = [t for t in self.terms[cid] if QUALIFIERS & set(_plain(t).split()) == wanted]
            if not terms:
                continue
            term, best = max(((t, similarity(name, t)) for t in terms), key=lambda x: (x[1], x[0]))
            if best >= min_score:
                scored.append((best, cid, term))
        # 4. Best score first (ties by concept id), truncated to top_k.
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self._match(cid, term, s, "fuzzy") for s, cid, term in scored[:top_k]]
