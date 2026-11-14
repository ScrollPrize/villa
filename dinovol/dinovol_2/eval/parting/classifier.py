"""Parting's classifier, its readout and the frozen gate (pure numpy plus scikit-learn).

A site gives two rows: (a, a2), the same layer (label 1), and (a, b), the next layer (label 0), at matched
distance. `pair_rows` turns the two points' features into one row that does not depend on the order of the two
points or on the sign of either normal: 669 columns from the groups tok, ct_n, ct_seg and p_tok and the two scalar
scores dino_base and ct_valley. The model is a soft vote (mean probability) of gradient boosting and a one-layer MLP,
trained on touching sites only.

The readout scores a run's sites once: per stratum, the model's AUC (a same-layer row scoring above a next-layer row,
ties half) and dino_base's on the same rows; on touching sites, 2.5 % bounds of the model's AUC and of its gain over
dino_base from 20,000 resamples of layer pairs (seed 0); and a permutation check, the mean AUC of 1,000 shuffled
labelings, which must sit within 0.5 plus or minus 3 standard errors. Model output only; nothing reads ink or text.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCALARS = ("dino_base", "ct_valley")
CONTROL = "dino_base"
LAYOUT = "parting-669"
COLUMNS = 669
RULE = {"stratum": "touching", "min_sites": 100, "min_pairs": 5, "auc": 0.75, "margin": 0.05, "delta_bound": 0.0,
        "pipeline_arm": "dino_base", "pipeline_stratum": "apart", "pipeline_auc": 0.85,
        "permutations": 1000, "permutation_sigmas": 3.0}
BOOT = 20000
STRATA = ("touching", "apart", "mixed", "fused")
# Known model files, by sha256: loading a joblib file runs pickled code, so score refuses any other file unless told
# to trust it (--trust-model). The released file is on Hugging Face (SharpMeow/parting-v1); fit makes your own.
KNOWN_MODELS = {
    "f0684ab2ec5725ad9050e08bcdf73df22983960ba906d222cb9c8068a0fdc601":
        "parting-v1: the frozen model of the Frag3 test (29 Oct 2026), trained on PHerc0139 and PHerc0172 "
        "windings; on Hugging Face as SharpMeow/parting-v1",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc(same, cross):
    """P(a same-layer score > a next-layer score), ties counted half (Mann-Whitney)."""
    same, cross = np.asarray(same, float), np.asarray(cross, float)
    if not len(same) or not len(cross):
        return float("nan")
    allv = np.concatenate([same, cross])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rs = ranks[:len(same)].sum()
    return float((rs - len(same) * (len(same) + 1) / 2) / (len(same) * len(cross)))


def fast_auc(s, x):
    """The same AUC by sorting, for the bootstrap loop."""
    x = np.sort(x)
    lt = np.searchsorted(x, s, side="left")
    le = np.searchsorted(x, s, side="right")
    return float((lt + 0.5 * (le - lt)).sum() / (len(s) * len(x)))


def load_run(folder):
    """(rows, groups) of a run folder: rows from results.jsonl (a resumed site's later row wins) in features.npz order,
    groups {name: float32 array} for the four feature groups."""
    folder = Path(folder)
    rows = {}
    for line in (folder / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    z = np.load(folder / "features.npz")
    ids = [int(i) for i in z["ids"]]
    missing = [g for g in ("tok", "ct_n", "ct_seg", "p_tok") if g not in z.files]
    if missing:
        raise SystemExit(f"{folder}/features.npz has no {', '.join(missing)}: not a Parting feature run")
    F = {k: z[k].astype(np.float32) for k in ("tok", "ct_n", "ct_seg", "p_tok")}
    keep = [n for n, i in enumerate(ids) if i in rows]
    return [rows[ids[n]] for n in keep], {k: v[keep] for k, v in F.items()}


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def _even_odd(p):
    """Even part and odd magnitude of a profile along a normal: the same whichever way the normal points."""
    r = p[..., ::-1]
    return (p + r) / 2, np.abs(p - r)[..., : p.shape[-1] // 2 + 1] / 2


def pair_rows(F, scores, j):
    """Rows for pair j (0: a with a2, 1: a with b) of every site, symmetric in the two points and free of normal sign.

    F: {group: (S, 3 or 2, ...)}; scores: (S, 2) of that pair's dino_base and ct_valley."""
    k = 1 + j
    cols = []
    u, v = _unit(F["tok"][:, 0]), _unit(F["tok"][:, k])
    cols += [u * v, np.abs(u - v)]
    eu, ou = _even_odd(F["ct_n"][:, 0])
    ev, ov = _even_odd(F["ct_n"][:, k])
    cols += [(eu + ev) / 2, np.abs(eu - ev), (ou + ov) / 2, np.abs(ou - ov)]
    s = F["ct_seg"][:, j]
    e, o = _even_odd(s)
    cols += [e, o, s.min(1, keepdims=True) - 0.5 * (s[:, :1] + s[:, -1:])]
    ca, cx = F["p_tok"][:, j, 0].astype(np.float32), F["p_tok"][:, j, 1].astype(np.float32)
    m = np.maximum(ca, cx)                                             # best link to either end at each sample
    h = m.shape[1] // 2
    cols += [m.min(1, keepdims=True), m.mean(1, keepdims=True), np.sort(m, axis=1)[:, :3],
             0.5 * (m + m[:, ::-1])[:, :h]]
    cols += [scores[:, n:n + 1] for n in range(len(SCALARS))]
    return np.concatenate(cols, axis=1).astype(np.float32)


def design(rows, F):
    """X with two rows per site (same pair then next-layer pair), y (1 same), layer pair and stratum per row, and
    dino_base's score per row."""
    sc = [np.array([[r["scores"][a][j] for a in SCALARS] for r in rows], np.float32) for j in (0, 1)]
    X0, X1 = pair_rows(F, sc[0], 0), pair_rows(F, sc[1], 1)
    n = len(rows)
    X = np.empty((2 * n, X0.shape[1]), np.float32)
    X[0::2], X[1::2] = X0, X1
    y = np.tile([1, 0], n)
    grp = np.repeat([r["pair"] for r in rows], 2)
    strat = np.repeat([r["stratum"] for r in rows], 2)
    ctrl = np.array([[r["scores"][CONTROL][0], r["scores"][CONTROL][1]] for r in rows]).ravel()
    return X, y, grp, strat, ctrl


def make():
    """The recipe: soft vote of HistGradientBoosting (600 iterations, 31 leaves) and an MLP (64 units)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    hgb = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                         random_state=0)
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500,
                                                        early_stopping=True, random_state=0))
    return VotingClassifier([("hgb", hgb), ("mlp", mlp)], voting="soft")


def load_model(path, trust=False):
    """(model, record) from a joblib bundle. Refuses a file whose sha256 is not a released one unless trust."""
    got = sha256(path)
    if got not in KNOWN_MODELS and not trust:
        raise SystemExit(f"{path}: sha256 {got} is not a known Parting model; loading it would run its pickled "
                         "code, so pass --trust-model only for a file you made or trust")
    import joblib
    bundle = joblib.load(path)
    cols = bundle.get("columns")
    n = int(np.asarray(cols).sum()) if cols is not None else bundle.get("features")
    if n != COLUMNS:
        raise SystemExit(f"{path}: the model reads {n} columns, not the {COLUMNS} of {LAYOUT}")
    return bundle["model"], {"file": Path(path).name, "sha256": got, "known": KNOWN_MODELS.get(got)}


def fit(dirs, out):
    """Fit the recipe on the touching sites of the given run folders and save it as a joblib bundle."""
    import joblib
    Xs, ys, ss = [], [], []
    names = [Path(d).name for d in dirs]
    if len(set(names)) != len(names):
        raise SystemExit("two --run folders share a name; copy one under another name")
    for d in dirs:
        rows, F = load_run(d)
        X, y, _, strat, _ = design(rows, F)
        Xs.append(X)
        ys.append(y)
        ss.append(strat)
    X, y, strat = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)
    tr = strat == "touching"
    if not tr.any():
        raise SystemExit("no touching sites in the given runs")
    m = make().fit(X[tr], y[tr])
    joblib.dump({"model": m, "choice": "vote|touching|parting", "features": COLUMNS, "columns": None,
                 "layout": LAYOUT, "runs": names}, out)
    return {"rows": int(tr.sum()), "runs": names, "model": str(out), "sha256": sha256(out)}


def boot_delta(s, c, y, grp, n=BOOT, seed=0):
    """2.5 % bounds of the model AUC and of model minus control, resampling layer pairs."""
    pairs = sorted(set(grp))
    idx = {p: np.nonzero(grp == p)[0] for p in pairs}
    rng = np.random.default_rng(seed)
    va, vd = [], []
    for _ in range(n):
        pick = np.concatenate([idx[pairs[i]] for i in rng.integers(0, len(pairs), len(pairs))])
        ys, ss, cs = y[pick], s[pick], c[pick]
        a = fast_auc(ss[ys == 1], ss[ys == 0])
        va.append(a)
        vd.append(a - fast_auc(cs[ys == 1], cs[ys == 0]))
    return float(np.percentile(va, 2.5)), float(np.percentile(vd, 2.5))


def perm_gate(s, y, k=1000, sigmas=3.0):
    """(ok, mean, half_width): the mean AUC of k label permutations (seeds 0 to k-1) against 0.5 plus or minus
    sigmas standard errors of that mean. It fails only on a broken AUC or permutation, never on the model."""
    s, y = np.asarray(s, float), np.asarray(y, int)
    a = np.array([auc(s[p == 1], s[p == 0]) for p in (np.random.default_rng(i).permutation(y) for i in range(k))])
    half = sigmas * a.std(ddof=1) / np.sqrt(k)
    m = float(a.mean())
    return bool(abs(m - 0.5) <= half), m, float(half)


def readout(rows, s, y, grp, strat, ctrl, boot=BOOT, permutations=None, rule=RULE, judge=True, sites=None):
    """The readout and, with judge, the verdict (PASS, FAIL or VOID) under rule. s: the model's score per row."""
    grp, strat = np.asarray(grp), np.asarray(strat)
    table = {}
    for st in STRATA:
        m = strat == st
        if not m.any():
            continue
        table[st] = {"sites": int(m.sum() // 2), "pairs": len(set(grp[m])),
                     "parting": round(auc(s[m & (y == 1)], s[m & (y == 0)]), 4),
                     CONTROL: round(auc(ctrl[m & (y == 1)], ctrl[m & (y == 0)]), 4)}
    if sites is not None:          # reported, not judged: touching sites whose larger normal gap is at most 4 voxels
        by = {x["id"]: x for x in sites}
        close = np.repeat([max(by[r["id"]]["gap_a"], by[r["id"]]["gap_b"]) <= 4 for r in rows], 2)
        close &= strat == "touching"
        if close.any():
            e = {"sites": int(close.sum() // 2), "pairs": len(set(grp[close])),
                 "parting": round(auc(s[close & (y == 1)], s[close & (y == 0)]), 4),
                 CONTROL: round(auc(ctrl[close & (y == 1)], ctrl[close & (y == 0)]), 4)}
            e["delta"] = round(e["parting"] - e[CONTROL], 4)
            table["touching_gap_le4"] = e
    R = rule
    m = strat == R["stratum"]
    t = table.get(R["stratum"], {"sites": 0, "pairs": 0, "parting": float("nan"), CONTROL: float("nan")})
    delta = t["parting"] - t[CONTROL]
    doc = {"rule": R, "table": table}
    if m.sum() >= 4 and len(set(grp[m])) >= 1:
        bound, dbound = boot_delta(s[m], ctrl[m], y[m], grp[m], boot)
    else:
        bound = dbound = float("nan")
    k = R["permutations"] if permutations is None else permutations
    ok, mean, half = perm_gate(s[m], y[m], k, R["permutation_sigmas"]) if m.any() else (False, float("nan"), 0.0)
    doc["primary"] = {"parting": t["parting"], CONTROL: t[CONTROL], "delta": round(delta, 4),
                      "parting_bound": round(bound, 4), "delta_bound": round(dbound, 4)}
    doc["permutation_gate"] = {"k": k, "sigmas": R["permutation_sigmas"], "mean": round(mean, 5),
                               "half_width": round(half, 5), "ok": ok}
    pairs = {}
    for g in sorted(set(grp[m])):
        q = m & (grp == g)
        a1, a0 = auc(s[q & (y == 1)], s[q & (y == 0)]), auc(ctrl[q & (y == 1)], ctrl[q & (y == 0)])
        pairs[str(g)] = {"sites": int(q.sum() // 2), "parting": round(a1, 4), CONTROL: round(a0, 4),
                         "delta": round(a1 - a0, 4)}
    doc["per_pair"] = pairs
    if not judge:
        doc["verdict"] = None
        return doc
    void = []
    if t["sites"] < R["min_sites"] or t["pairs"] < R["min_pairs"]:
        void.append(f"{t['sites']} {R['stratum']} sites over {t['pairs']} pairs")
    pipe = table.get(R["pipeline_stratum"], {}).get(R["pipeline_arm"], float("nan"))
    if not pipe >= R["pipeline_auc"]:
        void.append(f"pipeline check {R['pipeline_arm']} {pipe} on {R['pipeline_stratum']} < {R['pipeline_auc']}")
    if not ok:
        void.append(f"permutation mean AUC {mean:.4f} outside 0.5 +- {half:.4f}")
    passed = t["parting"] >= R["auc"] and delta >= R["margin"] and dbound > R["delta_bound"]
    doc["void_reasons"] = void
    doc["verdict"] = "VOID" if void else ("PASS" if passed else "FAIL")
    return doc


def without_pair(s, y, grp, strat, ctrl, pair, stratum="touching"):
    """Reported only: the model's and dino_base's AUC on the stratum with one layer pair left out."""
    m = (np.asarray(strat) == stratum) & (np.asarray(grp) != pair)
    if not m.any():
        return None
    b1, b0 = auc(s[m & (y == 1)], s[m & (y == 0)]), auc(ctrl[m & (y == 1)], ctrl[m & (y == 0)])
    return {"left_out": pair, "sites": int(m.sum() // 2), "parting": round(b1, 4), CONTROL: round(b0, 4),
            "delta": round(b1 - b0, 4)}
