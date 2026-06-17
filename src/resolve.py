"""
Entity resolution pipeline (supervised record linkage).

  1. BLOCKING       -- compare only records sharing a cheap key, turning an
                       O(n^2) all-pairs scan into something tractable.
  2. FEATURIZE      -- per candidate pair, build similarity features:
                       fuzzy name / email / address ratios PLUS two crisp
                       discriminative signals (same street number, same email
                       local-part) that separate look-alike different people.
  3. CLASSIFY       -- logistic regression learns the weights and decision
                       boundary from labeled pairs (Fellegi-Sunter style),
                       evaluated on a held-out split so the score is honest.
  4. CLUSTER        -- connected components of predicted matches = entities.
  5. EVALUATE       -- pairwise precision / recall / F1 vs ground truth.
"""
import json
import os
import re
from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def block_key(row):
    name = row["name"].lower().replace(" ", "")
    city = row["address"].split(",")[-1].strip().lower()
    return f"{name[:1]}{city[:2]}"


def street_number(addr):
    m = re.match(r"\s*(\d+)", addr)
    return m.group(1) if m else ""


def email_local(email):
    local = email.split("@")[0]
    return re.sub(r"[.\-+0-9]", "", local)        # normalize dots/tags/digits out


def features(a, b):
    return [
        fuzz.WRatio(a["name"], b["name"]) / 100,
        fuzz.token_sort_ratio(a["email"], b["email"]) / 100,
        fuzz.token_set_ratio(a["address"], b["address"]) / 100,
        1.0 if street_number(a["address"]) == street_number(b["address"]) else 0.0,
        1.0 if email_local(a["email"]) == email_local(b["email"]) else 0.0,
    ]


def candidate_pairs(df):
    df = df.copy()
    df["_block"] = df.apply(block_key, axis=1)
    pairs = []
    for _, grp in df.groupby("_block"):
        idx = grp["record_id"].tolist()
        pairs.extend(combinations(idx, 2))
    return pairs


def pairwise_prf(df, clusters):
    truth = dict(zip(df["record_id"], df["entity_id"]))
    by_cluster, by_entity = defaultdict(list), defaultdict(list)
    for r in df["record_id"]:
        by_cluster[clusters[r]].append(r)
        by_entity[truth[r]].append(r)

    def pset(groups):
        s = set()
        for m in groups.values():
            s.update(combinations(sorted(m), 2))
        return s

    pred, gold = pset(by_cluster), pset(by_entity)
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "true_pos": tp, "false_pos": fp, "false_neg": fn}


def main():
    df = pd.read_csv(f"{ROOT}/src/records.csv")
    recs = df.set_index("record_id").to_dict("index")
    truth = dict(zip(df["record_id"], df["entity_id"]))

    pairs = candidate_pairs(df)
    X = np.array([features(recs[i], recs[j]) for i, j in pairs])
    y = np.array([1 if truth[i] == truth[j] else 0 for i, j in pairs])

    # train/test split on pairs -- model never sees test pairs during fit
    idx = np.arange(len(pairs))
    tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=42)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X[tr], y[tr])

    # classifier metrics on held-out pairs
    from sklearn.metrics import precision_score, recall_score, f1_score
    pred_te = clf.predict(X[te])
    pair_metrics = {
        "precision": round(precision_score(y[te], pred_te), 3),
        "recall": round(recall_score(y[te], pred_te), 3),
        "f1": round(f1_score(y[te], pred_te), 3),
    }

    # cluster using ALL predicted matches
    pred_all = clf.predict(X)
    g = nx.Graph()
    g.add_nodes_from(df["record_id"])
    g.add_edges_from([pairs[k] for k in range(len(pairs)) if pred_all[k] == 1])
    clusters = {}
    for cid, comp in enumerate(nx.connected_components(g)):
        for r in comp:
            clusters[r] = cid

    cluster_metrics = pairwise_prf(df, clusters)
    naive = len(df) * (len(df) - 1) // 2
    report = {
        "pair_classifier_heldout": pair_metrics,
        "end_to_end_clustering": cluster_metrics,
        "records": len(df),
        "true_entities": int(df.entity_id.nunique()),
        "resolved_entities": len(set(clusters.values())),
        "candidate_pairs_after_blocking": len(pairs),
        "pairs_without_blocking": naive,
        "blocking_reduction_pct": round((1 - len(pairs) / naive) * 100, 1),
        "learned_weights": dict(zip(
            ["name", "email", "address", "same_street_no", "same_email_local"],
            [round(float(w), 2) for w in clf.coef_[0]])),
    }
    os.makedirs(f"{ROOT}/artifacts", exist_ok=True)
    with open(f"{ROOT}/artifacts/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
