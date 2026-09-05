import json
import random

ATTACKS = "data/attacks.json"
HARD_NEGATIVES = "data/hard_negatives.json"
HOLDOUT_FRACTION = 0.25
SPLIT_SEED = 20260903


def holdout_ids() -> set[str]:
    attacks = json.load(open(ATTACKS, encoding="utf-8"))
    strata: dict[tuple[str, str], list[str]] = {}
    for attack in attacks:
        strata.setdefault((attack["source"], attack["goal"]), []).append(attack["id"])
    prng = random.Random(SPLIT_SEED)
    reserved = set()
    for stratum in sorted(strata):
        ids = sorted(strata[stratum])
        prng.shuffle(ids)
        reserved.update(ids[:round(len(ids) * HOLDOUT_FRACTION)])
    return reserved


def load_attacks(split: str) -> list[dict]:
    reserved = holdout_ids()
    attacks = json.load(open(ATTACKS, encoding="utf-8"))
    return [a for a in attacks if (a["id"] in reserved) == (split == "holdout")]


def load_hard_negatives() -> list[dict]:
    return json.load(open(HARD_NEGATIVES, encoding="utf-8"))
