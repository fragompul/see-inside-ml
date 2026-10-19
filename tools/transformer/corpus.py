"""Synthetic mini-story corpus for the tiny GPT.

Why synthetic? It needs no download or licence, and the grammar is rich enough that a small
transformer has to learn *interpretable* things: subject-verb agreement, pronoun gender,
`it` -> object, `they` -> two names, and copying a name seen a few tokens earlier. Those are
exactly the behaviours the attention heatmaps in the web app let you see.

Every story is a few short sentences about one protagonist, so facts must be carried across
sentences (who? which object? which place?).
"""

from __future__ import annotations

import random

FEMALE = ["Anna", "Mia", "Eva", "Lily", "Nora", "Zoe", "Ruby", "Ella"]
MALE = ["Tom", "Leo", "Sam", "Max", "Ben", "Jack", "Finn", "Oscar"]

# singular noun -> plural noun
ANIMALS = {
    "cat": "cats",
    "dog": "dogs",
    "bird": "birds",
    "fox": "foxes",
    "mouse": "mice",
    "horse": "horses",
    "rabbit": "rabbits",
    "duck": "ducks",
}
OBJECTS = ["ball", "book", "apple", "key", "hat", "cake", "kite", "lamp"]
PLACES = ["park", "garden", "kitchen", "school", "river", "forest", "farm", "beach"]
ADJECTIVES = ["big", "small", "red", "happy", "tired", "hungry", "quiet", "quick", "kind", "brave"]

# (third person singular, base form) pairs
TRANSITIVE = [
    ("sees", "see"),
    ("finds", "find"),
    ("likes", "like"),
    ("takes", "take"),
    ("drops", "drop"),
    ("opens", "open"),
    ("carries", "carry"),
    ("follows", "follow"),
]
INTRANSITIVE = [
    ("runs", "run"),
    ("sleeps", "sleep"),
    ("jumps", "jump"),
    ("sings", "sing"),
    ("waits", "wait"),
    ("plays", "play"),
]
PAST = {
    "finds": "found",
    "takes": "took",
    "sees": "saw",
    "likes": "liked",
    "drops": "dropped",
    "opens": "opened",
    "carries": "carried",
    "follows": "followed",
}


def _article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def _person(rng: random.Random):
    if rng.random() < 0.5:
        return rng.choice(FEMALE), "she", "her"
    return rng.choice(MALE), "he", "his"


def _story_person(rng: random.Random) -> str:
    """Protagonist finds an object and carries it somewhere (pronoun + `it` coreference)."""
    name, pron, poss = _person(rng)
    obj, place = rng.choice(OBJECTS), rng.choice(PLACES)
    verb = rng.choice(["finds", "takes", "sees"])
    s = [f"{name} {PAST[verb]} {_article(obj)} {obj}."]
    tail = rng.choice(
        [
            f"{pron.capitalize()} took it to the {place}.",
            f"{pron.capitalize()} liked it a lot.",
            f"{pron.capitalize()} carried it to the {place}.",
        ]
    )
    s.append(tail)
    if rng.random() < 0.5:
        s.append(f"At the {place}, {name} was happy.")
    return " ".join(s)


def _story_pair(rng: random.Random) -> str:
    """Two friends: `They` refers back to both names."""
    a, b = rng.sample(FEMALE + MALE, 2)
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    s = [f"{a} and {b} went to the {place}.", f"They found {_article(obj)} {obj} there."]
    if rng.random() < 0.6:
        s.append(f"{a} said hello to {b}.")
        s.append(f"{b} said hello to {a}.")
    return " ".join(s)


def _story_animal(rng: random.Random) -> str:
    """Animal stories use singular/plural agreement and `It` / `They`."""
    animal = rng.choice(list(ANIMALS))
    adj = rng.choice(ADJECTIVES)
    place = rng.choice(PLACES)
    if rng.random() < 0.55:
        v_s, _ = rng.choice(INTRANSITIVE)
        obj = rng.choice(OBJECTS)
        t_s, _ = rng.choice(TRANSITIVE)
        parts = [
            f"The {adj} {animal} {v_s} in the {place}.",
            f"It {t_s} the {obj}.",
        ]
        if rng.random() < 0.5:
            parts.append(f"The {animal} is {rng.choice(ADJECTIVES)}.")
        return " ".join(parts)
    plural = ANIMALS[animal]
    _, v_b = rng.choice(INTRANSITIVE)
    _, t_b = rng.choice(TRANSITIVE)
    obj = rng.choice(OBJECTS)
    parts = [
        f"The {adj} {plural} {v_b} in the {place}.",
        f"They {t_b} the {obj}.",
    ]
    if rng.random() < 0.5:
        parts.append(f"The {plural} are {rng.choice(ADJECTIVES)}.")
    return " ".join(parts)


def _story_list(rng: random.Random) -> str:
    """Three friends and something they all like: a pattern to copy."""
    names = rng.sample(FEMALE + MALE, 3)
    obj = rng.choice(OBJECTS)
    place = rng.choice(PLACES)
    return (
        f"{names[0]}, {names[1]} and {names[2]} like {obj}s. "
        f"{names[0]} brings a {obj} to the {place}."
    )


STORIES = [(_story_person, 0.34), (_story_pair, 0.22), (_story_animal, 0.30), (_story_list, 0.14)]


def story(rng: random.Random) -> str:
    r = rng.random()
    acc = 0.0
    for fn, weight in STORIES:
        acc += weight
        if r < acc:
            return fn(rng)
    return STORIES[-1][0](rng)


def generate(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [story(rng) for _ in range(n)]


if __name__ == "__main__":
    for s in generate(8, seed=1):
        print(s)
