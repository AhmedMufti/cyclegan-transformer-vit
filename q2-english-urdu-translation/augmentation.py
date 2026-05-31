"""
Lightweight NMT data augmentation, following Wei & Zou (2019)
"Easy Data Augmentation" (EDA):

  - random_deletion : drop each token with probability p
  - random_swap     : swap N pairs of adjacent tokens
  - random_synonym  : (optional, requires nltk.wordnet) replace words with synonyms

Used only on the SOURCE (English) side. The target (Urdu) is left untouched
because the target label must exactly match what the model is asked to
generate.

At training time each sample is augmented with probability `p_aug`;
otherwise the original English is used. Validation/test sentences are
never augmented.
"""
import random
from typing import List

# Try to enable WordNet synonym replacement if nltk is present.
_HAS_WORDNET = False
try:
    import nltk  # noqa
    from nltk.corpus import wordnet as _wn
    # silent download
    try:
        _wn.synsets("example")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    _HAS_WORDNET = True
except Exception:
    _HAS_WORDNET = False


def random_deletion(tokens: List[str], p: float = 0.1) -> List[str]:
    """Drop each token independently with probability p. Keeps at least one."""
    if len(tokens) <= 1:
        return tokens
    kept = [t for t in tokens if random.random() > p]
    return kept if kept else [random.choice(tokens)]


def random_swap(tokens: List[str], n: int = 1) -> List[str]:
    """Swap n pairs of adjacent tokens (in place copy)."""
    tokens = list(tokens)
    for _ in range(n):
        if len(tokens) < 2:
            break
        i = random.randint(0, len(tokens) - 2)
        tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
    return tokens


def _synonyms(word: str):
    if not _HAS_WORDNET:
        return []
    syns = set()
    for syn in _wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                syns.add(name)
    return list(syns)


def random_synonym(tokens: List[str], n: int = 1) -> List[str]:
    """Replace up to n tokens with a WordNet synonym (no-op if WordNet unavailable)."""
    if not _HAS_WORDNET:
        return tokens
    tokens = list(tokens)
    candidates = [i for i, t in enumerate(tokens) if t.isalpha()]
    random.shuffle(candidates)
    swapped = 0
    for i in candidates:
        syns = _synonyms(tokens[i])
        if syns:
            tokens[i] = random.choice(syns)
            swapped += 1
            if swapped >= n:
                break
    return tokens


def augment_sentence(sentence: str, p_aug: float = 0.5,
                     ops=("deletion", "swap", "synonym")) -> str:
    """With prob p_aug, apply one random op from `ops`; else return unchanged."""
    if random.random() > p_aug:
        return sentence
    tokens = sentence.split()
    if len(tokens) < 2:
        return sentence
    op = random.choice(ops)
    if op == "deletion":
        tokens = random_deletion(tokens, p=0.1)
    elif op == "swap":
        tokens = random_swap(tokens, n=max(1, len(tokens) // 20))
    elif op == "synonym":
        tokens = random_synonym(tokens, n=max(1, len(tokens) // 10))
    return " ".join(tokens)


if __name__ == "__main__":
    random.seed(0)
    for s in ["how are you today?",
              "i love reading books in the library",
              "the weather is very nice this morning"]:
        print("orig:", s)
        for _ in range(3):
            print("  ->", augment_sentence(s, p_aug=1.0))
        print()
