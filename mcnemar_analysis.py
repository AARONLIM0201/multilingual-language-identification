import json
import os
from collections import OrderedDict

try:
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.proportion import proportion_confint
except Exception:
    mcnemar = None
    proportion_confint = None

try:
    from scipy.stats import binom_test
except Exception:
    binom_test = None


def wilson_ci(k, n, alpha=0.05):
    # try statsmodels first
    if proportion_confint is not None:
        low, high = proportion_confint(k, n, alpha=alpha, method='wilson')
        return low, high
    # fallback to simple approximation
    import math
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z = 1.96  # approx for 95%
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    adj = z * math.sqrt((p*(1-p)+z*z/(4*n))/n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return max(0.0, low), min(1.0, high)


def exact_mcnemar_p(b, c):
    # two-sided exact binomial test on min(b,c)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if binom_test is not None:
        try:
            return binom_test(k, n, p=0.5, alternative='two-sided')
        except TypeError:
            # older scipy signature
            return binom_test(k, n, 0.5)
    # fallback: use simple two-sided p from cumulative
    from math import comb
    # compute p-value as two-sided by summing probabilities as or more extreme
    probs = []
    for i in range(0, n+1):
        probs.append((i, comb(n, i) * (0.5**n)))
    # two-sided p: sum probs with prob <= prob(obs)
    obs_prob = comb(n, k) * (0.5**n)
    pval = sum(p for i, p in probs if p <= obs_prob + 1e-15)
    return min(1.0, pval)


def chi2_mcnemar(table):
    # use statsmodels if available
    if mcnemar is not None:
        res = mcnemar(table, exact=False)
        return res.statistic, float(res.pvalue)
    # otherwise compute chi-square approximation without continuity correction
    b = table[0][1]
    c = table[1][0]
    denom = b + c
    if denom == 0:
        return 0.0, 1.0
    stat = (b - c)**2 / denom
    # p-value from chi2(1)
    try:
        from scipy.stats import chi2
        p = chi2.sf(stat, 1)
    except Exception:
        p = 1.0
    return stat, float(p)


def analyze(json_path, out_path=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        j = json.load(f)

    labels = j['labels_and_predictions']['true_labels']
    preds = j['labels_and_predictions']['predictions']
    methods = ['Audio Only (Wav2Vec2)', 'Text Only (DistilBERT)', 'Equal Weighting (0.5/0.5)',
               'Accuracy-Weighted Fusion', 'Confidence-Adaptive Fusion']

    # ensure arrays lengths
    n = len(labels)
    results = OrderedDict()

    # compute per-method correctness
    correct = {}
    for m in methods:
        if m not in preds:
            raise KeyError(f"Method {m} not found in predictions")
        if len(preds[m]) != n:
            raise ValueError(f"Length mismatch for {m}: {len(preds[m])} vs {n}")
        correct[m] = [1 if preds[m][i] == labels[i] else 0 for i in range(n)]

    ref = 'Equal Weighting (0.5/0.5)'
    baselines = ['Audio Only (Wav2Vec2)', 'Text Only (DistilBERT)', 'Confidence-Adaptive Fusion', 'Accuracy-Weighted Fusion']

    for bname in baselines:
        b_correct = correct[ref]
        o_correct = correct[bname]
        # contingency cells for McNemar
        # table: [[n00, n01],[n10,n11]] where rows are ref correct/wrong? We will use table = [[ref_correct & other_correct? Wait]]
        # Standard McNemar uses table = [[both correct? not used], [ref correct other wrong], [ref wrong other correct], [both wrong]]
        # We'll compute b = ref_correct & other_wrong; c = ref_wrong & other_correct
        b = sum(1 for i in range(n) if b_correct[i] == 1 and o_correct[i] == 0)
        c = sum(1 for i in range(n) if b_correct[i] == 0 and o_correct[i] == 1)
        # mcnemar table for statsmodels: [[n11, n10],[n01,n00]]? statsmodels expects [[a, b],[c, d]] where a = both successes.
        a = sum(1 for i in range(n) if b_correct[i] == 1 and o_correct[i] == 1)
        d = sum(1 for i in range(n) if b_correct[i] == 0 and o_correct[i] == 0)
        table = [[a, b], [c, d]]

        chi2_stat, chi2_p = chi2_mcnemar(table)
        exact_p = exact_mcnemar_p(b, c)

        # accuracies and CIs
        ref_acc = sum(b_correct)
        other_acc = sum(o_correct)
        ref_ci_low, ref_ci_high = wilson_ci(ref_acc, n)
        other_ci_low, other_ci_high = wilson_ci(other_acc, n)

        results[bname] = {
            'n': n,
            'b_ref_correct_other_wrong': int(b),
            'c_ref_wrong_other_correct': int(c),
            'chi2_stat': float(chi2_stat),
            'chi2_pvalue': float(chi2_p),
            'exact_pvalue': float(exact_p),
            'ref_accuracy_count': int(ref_acc),
            'other_accuracy_count': int(other_acc),
            'ref_accuracy': ref_acc / n,
            'other_accuracy': other_acc / n,
            'ref_wilson_ci_95': [ref_ci_low, ref_ci_high],
            'other_wilson_ci_95': [other_ci_low, other_ci_high]
        }

    if out_path is None:
        out_path = os.path.join(os.path.dirname(json_path), 'mcnemar_results.json')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # also print concise summary
    for k,v in results.items():
        print(f"Comparison: EqualWeight vs {k}")
        print(f" n={v['n']}, b={v['b_ref_correct_other_wrong']}, c={v['c_ref_wrong_other_correct']}")
        print(f" chi2_stat={v['chi2_stat']:.4f}, chi2_p={v['chi2_pvalue']:.4g}")
        print(f" exact_p={v['exact_pvalue']:.4g}")
        print(f" EqualWeight acc={v['ref_accuracy']*100:.3f}% CI95={v['ref_wilson_ci_95']}")
        print(f" {k} acc={v['other_accuracy']*100:.3f}% CI95={v['other_wilson_ci_95']}\n")

    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='wav2vec2_distilbert_full_analysis_results.json')
    parser.add_argument('--output', '-o', default='mcnemar_results.json')
    args = parser.parse_args()
    analyze(args.input, args.output)
