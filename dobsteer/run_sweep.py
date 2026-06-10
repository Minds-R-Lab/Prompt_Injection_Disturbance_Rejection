"""Foundation-locking sweep: seed stability + target-FPR ROC for the gated
detect-then-cancel pipeline (Phase 3), efficiently.

Loads the model ONCE. For each seed it shuffles the train/test split, fits the
content-aware subspace+detector, and generates each attack prompt twice (base
+ gated) -- caching success flags and the gate score. Every target-FPR
threshold is then evaluated from the cache (no regeneration), giving the
ASR / FPR / TPR / benign-KL ROC for free. Aggregates mean +/- std over seeds.

  python -m dobsteer.run_sweep --model Qwen/Qwen2.5-7B-Instruct --device cuda \
      --seeds 0,1,2,3,4 --fprs 0,0.01,0.02,0.05,0.1 --k 32 --alpha 3 --tau 1.0
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import torch

from .prompts import BASE_PROMPTS, INJECTIONS, place_mid
from .run_phase2 import SUCCESS_MARKERS, Canceller, make_fmt, generate, next_logits
from .run_phase3 import template_span, fit_shared, gate_score, roc_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--fprs", default="0,0.01,0.02,0.05,0.1")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--max-test", type=int, default=20)
    ap.add_argument("--use-data", action="store_true", help="draw benign from data.load_benign")
    ap.add_argument("--n-benign", type=int, default=300)
    ap.add_argument("--out", default="sweep_results.json")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    fprs = [float(x) for x in args.fprs.split(",")]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device).eval()
    fmt = make_fmt(tok)
    span = template_span(tok, fmt)
    L = model.config.num_hidden_layers
    all_layers = list(range(L))
    det_layers = list(range(L // 4, 3 * L // 4))
    if args.use_data:
        from . import data
        POOL = data.load_benign(n=args.n_benign, seed=999)
    else:
        POOL = BASE_PROMPTS
    n = len(POOL)
    print(f"Model={args.model} seeds={seeds} fprs={fprs} k={args.k} "
          f"alpha={args.alpha} tau={args.tau} span={span}")

    per_seed = []   # list of dict: seed -> {fpr -> metrics, gate_auc, base_asr}
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        prompts = [POOL[i] for i in perm]
        n_tr = n // 3
        n_ca = n // 3
        train_p = prompts[:n_tr]
        calib_p = prompts[n_tr:n_tr + n_ca]          # held-out for threshold
        test_p = prompts[n_tr + n_ca:][: args.max_test]

        V, mu, sd, mbar, msd = fit_shared(model, tok, train_p, fmt, args.device,
                                          args.k, all_layers, span)
        gs = lambda text: gate_score(model, tok, text, args.device,
                                     V, mu, sd, mbar, msd, det_layers, span)

        bca = torch.tensor([gs(fmt(p)) for p in calib_p])   # held-out benign for threshold

        # cache test attacks: (j, score, success_base, success_gated)
        atk = []
        for j, inj in enumerate(INJECTIONS):
            for placement in ("append", "mid"):
                for p in test_p:
                    q = p + inj if placement == "append" else place_mid(p, inj)
                    score = gs(fmt(q))
                    b = generate(model, tok, fmt(q), args.device, args.max_new)
                    ctl = Canceller(V, mu, sd, mbar, msd, args.tau, args.alpha).attach(model)
                    d = generate(model, tok, fmt(q), args.device, args.max_new)
                    ctl.detach()
                    sb = any(m in b for m in SUCCESS_MARKERS[j])
                    sg = any(m in d for m in SUCCESS_MARKERS[j])
                    atk.append((score, sb, sg))
        # cache benign: (score, kl_gated)
        ben = []
        for p in test_p:
            score = gs(fmt(p))
            lp = torch.log_softmax(next_logits(model, tok, fmt(p), args.device), -1)
            ctl = Canceller(V, mu, sd, mbar, msd, args.tau, args.alpha).attach(model)
            lq = torch.log_softmax(next_logits(model, tok, fmt(p), args.device), -1)
            ctl.detach()
            ben.append((score, float((lp.exp() * (lp - lq)).sum())))

        atk_scores = [a[0] for a in atk]; ben_scores = [b[0] for b in ben]
        gate_auc = roc_auc(atk_scores, ben_scores)
        base_asr = sum(a[1] for a in atk) / len(atk)

        rec = {"seed": seed, "gate_auc": gate_auc, "base_asr": base_asr, "fpr": {}}
        for fpr in fprs:
            thr = torch.quantile(bca, 1.0 - fpr).item() if fpr > 0 else bca.max().item()
            test_fpr = sum(s > thr for s, _ in ben) / len(ben)
            tpr = sum(s > thr for s, _, _ in atk) / len(atk)
            asr = sum((sg if s > thr else sb) for s, sb, sg in atk) / len(atk)
            bkl = sum((kl if s > thr else 0.0) for s, kl in ben) / len(ben)
            rec["fpr"][fpr] = {"thr": thr, "test_fpr": test_fpr, "tpr": tpr,
                               "asr": asr, "benign_kl": bkl}
        per_seed.append(rec)
        print(f"  seed {seed}: gate AUC={gate_auc:.3f} base ASR={base_asr:.2f}")

    # aggregate
    def agg(key, fpr):
        xs = [r["fpr"][fpr][key] for r in per_seed]
        return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)

    aucs = [r["gate_auc"] for r in per_seed]
    print(f"\n  Gate AUC over {len(seeds)} seeds: {st.mean(aucs):.3f} "
          f"+/- {st.pstdev(aucs) if len(aucs)>1 else 0.0:.3f}")
    print(f"  Base ASR: {st.mean([r['base_asr'] for r in per_seed]):.2f}")
    print(f"\n  {'targetFPR':>9} {'testFPR':>14} {'TPR':>14} {'ASR':>14} {'benignKL':>16}")
    for fpr in fprs:
        f_m, f_s = agg("test_fpr", fpr); t_m, t_s = agg("tpr", fpr)
        a_m, a_s = agg("asr", fpr); k_m, k_s = agg("benign_kl", fpr)
        print(f"  {fpr:>9.2f} {f_m:>6.2f}+/-{f_s:<4.2f} {t_m:>6.2f}+/-{t_s:<4.2f} "
              f"{a_m:>6.2f}+/-{a_s:<4.2f} {k_m:>8.4f}+/-{k_s:<6.4f}")

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "seeds": seeds, "fprs": fprs,
                   "k": args.k, "alpha": args.alpha, "tau": args.tau,
                   "per_seed": per_seed}, f, indent=1)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
