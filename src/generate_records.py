"""
Generate a customer table riddled with duplicates -- the real-world mess
entity resolution exists to clean up.

Each TRUE entity has a stable identity (name, email handle, street address).
It appears 1-3 times with realistic corruptions of that identity:
  - name typos / transpositions / nicknames (Robert -> Bob)
  - email format drift (dots, plus-tags, provider typos) on the SAME handle
  - address abbreviations (Street -> St) and apartment formatting

Crucially, different entities get DIFFERENT handles and addresses, so the
matcher must use those signals to tell apart two people with the same name.
A ground-truth `entity_id` column lets us score precisely.
"""
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)
HERE = os.path.dirname(os.path.abspath(__file__))

FIRST = ["Robert", "Jennifer", "Michael", "Sarah", "David", "Emily", "James",
         "Jessica", "John", "Ashley", "Daniel", "Amanda", "Christopher",
         "Melissa", "Matthew", "Stephanie", "Joshua", "Nicole", "Andrew", "Laura"]
NICK = {"Robert": "Bob", "Michael": "Mike", "James": "Jim", "Jennifer": "Jen",
        "Jessica": "Jess", "Daniel": "Dan", "Christopher": "Chris", "Matthew": "Matt"}
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Okeke", "Obazee", "Anderson",
        "Taylor", "Thomas", "Moore", "Jackson", "White", "Harris", "Clark"]
STREETS = ["Maple Street", "Oak Avenue", "Cedar Lane", "Pine Road", "Elm Boulevard",
           "Birch Way", "Willow Drive", "Aspen Court"]
CITIES = ["Houston", "Dallas", "Austin", "Phoenix", "Denver"]

N_ENTITIES = 1500


def typo(s):
    if len(s) < 4 or RNG.random() > 0.45:
        return s
    i = RNG.integers(0, len(s) - 1)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]          # transpose two chars


def perturb_email(handle):
    """Same handle, drifting presentation."""
    provider_drift = RNG.choice(["gmail.com", "gmail.con", "gmail.com", "gmail.com"])
    h = handle
    if RNG.random() < 0.4:                              # insert/remove a dot
        h = h.replace(".", "") if "." in h else h
    tag = f"+{RNG.integers(1,99)}" if RNG.random() < 0.25 else ""
    return f"{h}{tag}@{provider_drift}"


def perturb_address(street, num, city):
    s = street.replace("Street", "St").replace("Avenue", "Ave").replace("Boulevard", "Blvd") \
        if RNG.random() < 0.5 else street
    apt = f" Apt {RNG.integers(1,30)}" if RNG.random() < 0.3 else ""
    return f"{num} {s}{apt}, {city}"


def generate():
    rows = []
    rec_id = 0
    for eid in range(N_ENTITIES):
        first = RNG.choice(FIRST)
        last = RNG.choice(LAST)
        num = int(RNG.integers(100, 9999))             # unique-ish per entity
        street = RNG.choice(STREETS)
        city = RNG.choice(CITIES)
        handle = f"{first.lower()}.{last.lower()}{eid}"  # entity-unique email handle
        n_dupes = RNG.choice([1, 2, 3], p=[0.55, 0.3, 0.15])
        for _ in range(n_dupes):
            fn = NICK.get(first, first) if RNG.random() < 0.25 else typo(first)
            ln = typo(last)
            rows.append({
                "record_id": rec_id,
                "entity_id": eid,                      # ground truth
                "name": f"{fn} {ln}",
                "email": perturb_email(handle),
                "address": perturb_address(street, num, city),
            })
            rec_id += 1
    return pd.DataFrame(rows).sample(frac=1, random_state=3).reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv(f"{HERE}/records.csv", index=False)
    dupes = len(df) - df.entity_id.nunique()
    print(f"{len(df):,} records · {df.entity_id.nunique()} true entities · "
          f"{dupes:,} duplicate records to resolve")
