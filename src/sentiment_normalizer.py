"""Conservative normalization for noisy sentiment text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z0-9]+(?:'[A-Za-z]+)?)(?![A-Za-z0-9])"
)

# These are reusable token or short-phrase mappings, not complete test sentences.
PHRASE_REPLACEMENTS = {
    r"(?<!\w)w/o(?!\w)": "without",
    r"(?<!\w)w/(?!\w)": "with",
    r"(?<!\w)b/c(?!\w)": "because",
}

COMMON_ABBREVIATIONS = {
    "ain't": "is not",
    "aint": "is not",
    "aren't": "are not",
    "arent": "are not",
    "bc": "because",
    "bcoz": "because",
    "bcz": "because",
    "btw": "by the way",
    "can't": "can not",
    "cant": "can not",
    "couldn't": "could not",
    "couldnt": "could not",
    "crs": "course",
    "cuz": "because",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "dont": "do not",
    "dunno": "do not know",
    "gonna": "going to",
    "gotta": "have to",
    "hadn't": "had not",
    "hadnt": "had not",
    "hasn't": "has not",
    "hasnt": "has not",
    "haven't": "have not",
    "havent": "have not",
    "idk": "i do not know",
    "imo": "in my opinion",
    "imho": "in my honest opinion",
    "im": "i am",
    "i'm": "i am",
    "ive": "i have",
    "i've": "i have",
    "isn't": "is not",
    "isnt": "is not",
    "kinda": "somewhat",
    "ngl": "not going to lie",
    "ppl": "people",
    "pls": "please",
    "plz": "please",
    "rn": "right now",
    "shouldn't": "should not",
    "shouldnt": "should not",
    "sorta": "somewhat",
    "tbh": "to be honest",
    "thats": "that is",
    "that's": "that is",
    "theres": "there is",
    "there's": "there is",
    "theyre": "they are",
    "they're": "they are",
    "tho": "though",
    "wasn't": "was not",
    "wasnt": "was not",
    "weren't": "were not",
    "werent": "were not",
    "we're": "we are",
    "won't": "will not",
    "wont": "will not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "youre": "you are",
    "you're": "you are",
    "y'all": "you all",
    "yall": "you all",
}

# Short, irregular, or high-frequency noisy forms are safer as exact mappings.
COMMON_SENTIMENT_MISSPELLINGS = {
    "amazn": "amazing",
    "awfull": "awful",
    "awsm": "awesome",
    "baad": "bad",
    "badd": "bad",
    "booring": "boring",
    "cleer": "clear",
    "confuzing": "confusing",
    "diffcult": "difficult",
    "difficlut": "difficult",
    "excelnt": "excellent",
    "gret": "great",
    "gr8": "great",
    "gud": "good",
    "h8": "hate",
    "helpfull": "helpful",
    "horrble": "horrible",
    "luv": "love",
    "realy": "really",
    "recomend": "recommend",
    "recommnd": "recommend",
    "sux": "sucks",
    "terrble": "terrible",
    "uncler": "unclear",
    "usefull": "useful",
    "useles": "useless",
}

SENTIMENT_SPELLING_TARGETS = frozenset(
    {
        "amazing",
        "awesome",
        "awful",
        "bad",
        "boring",
        "broken",
        "clear",
        "confusing",
        "difficult",
        "disappointing",
        "engaging",
        "excellent",
        "fantastic",
        "frustrating",
        "good",
        "great",
        "hard",
        "helpful",
        "horrible",
        "impossible",
        "impressive",
        "love",
        "organized",
        "outdated",
        "practical",
        "poor",
        "recommend",
        "recommended",
        "stressful",
        "terrible",
        "hate",
        "unclear",
        "unhelpful",
        "useful",
        "useless",
        "worthwhile",
        "sucks",
    }
)


@dataclass(frozen=True)
class SentimentReplacement:
    original: str
    replacement: str
    kind: str


@dataclass(frozen=True)
class SentimentNormalization:
    text: str
    replacements: tuple[SentimentReplacement, ...]


def _apply_case(original: str, replacement: str) -> str:
    if not original:
        return replacement
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _is_one_insertion_or_deletion(source: str, target: str) -> bool:
    if abs(len(source) - len(target)) != 1:
        return False
    shorter, longer = (
        (source, target) if len(source) < len(target) else (target, source)
    )
    short_index = 0
    long_index = 0
    skipped = False
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        if skipped:
            return False
        skipped = True
        long_index += 1
    return True


def _is_one_adjacent_transposition(source: str, target: str) -> bool:
    if len(source) != len(target) or len(source) < 7:
        return False
    differences = [
        index
        for index, (left, right) in enumerate(zip(source, target))
        if left != right
    ]
    if len(differences) != 2:
        return False
    first, second = differences
    return (
        second == first + 1
        and source[first] == target[second]
        and source[second] == target[first]
    )


def _is_inflection_variant(source: str, target: str) -> bool:
    return (
        source == target + "s"
        or source == target + "es"
        or source == target + "ed"
        or source == target + "d"
        or target == source + "s"
        or target == source + "es"
        or target == source + "ed"
        or target == source + "d"
    )


def correct_sentiment_spelling(token: str) -> str | None:
    """Return a unique conservative correction for a sentiment word."""

    normalized = token.lower()
    if not normalized.isalpha() or normalized in SENTIMENT_SPELLING_TARGETS:
        return None

    collapsed_candidates = {
        re.sub(r"(.)\1{2,}", r"\1", normalized),
        re.sub(r"(.)\1{2,}", r"\1\1", normalized),
    }
    collapsed_matches = collapsed_candidates & SENTIMENT_SPELLING_TARGETS
    if len(collapsed_matches) == 1:
        return next(iter(collapsed_matches))

    if len(normalized) < 5:
        return None

    candidates = [
        target
        for target in SENTIMENT_SPELLING_TARGETS
        if len(target) >= 5
        and normalized[:2] == target[:2]
        and not _is_inflection_variant(normalized, target)
        and (
            _is_one_insertion_or_deletion(normalized, target)
            or _is_one_adjacent_transposition(normalized, target)
        )
    ]
    return candidates[0] if len(candidates) == 1 else None


def normalize_sentiment_text_with_details(text: object) -> SentimentNormalization:
    if text is None:
        return SentimentNormalization("", ())

    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("’", "'").replace("‘", "'")
    replacements: list[SentimentReplacement] = []

    for pattern, replacement in PHRASE_REPLACEMENTS.items():
        def replace_phrase(match: re.Match[str]) -> str:
            original = match.group(0)
            replacements.append(
                SentimentReplacement(original, replacement, "abbreviation")
            )
            return _apply_case(original, replacement)

        value = re.sub(pattern, replace_phrase, value, flags=re.IGNORECASE)

    def replace_token(match: re.Match[str]) -> str:
        original = match.group(1)
        normalized = original.lower()
        replacement = COMMON_ABBREVIATIONS.get(normalized)
        kind = "abbreviation"
        if replacement is None:
            replacement = COMMON_SENTIMENT_MISSPELLINGS.get(normalized)
            kind = "spelling"
        if replacement is None:
            replacement = correct_sentiment_spelling(normalized)
            kind = "spelling"
        if replacement is None or replacement == normalized:
            return original

        cased_replacement = _apply_case(original, replacement)
        replacements.append(
            SentimentReplacement(original, cased_replacement, kind)
        )
        return cased_replacement

    value = TOKEN_PATTERN.sub(replace_token, value)
    value = re.sub(r"\s+", " ", value).strip()
    return SentimentNormalization(value, tuple(replacements))


def normalize_sentiment_text(text: object) -> str:
    return normalize_sentiment_text_with_details(text).text
