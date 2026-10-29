#!/usr/bin/env python3
import json, os, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = '/home/lied/Documents/Codex/2026-10-29/read/work'
MODEL_DIR = os.path.join(WORK, 'model')
OUT_DIR = os.path.join(WORK, 'evidence')
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    with open(os.path.join(WORK, 'scripts', 'corpus.json')) as f:
        corpus = json.load(f)
    print('loading tokenizer', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    print('loading model on GPU', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map='cuda',
        attn_implementation='sdpa'
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_inter = model.config.intermediate_size
    print(f'layers={n_layers} inter={n_inter}', flush=True)

    captures = {}
    hooks = []
    for L in range(n_layers):
        def make_hook(L):
            def fn(module, args, output):
                captures[L] = args[0].detach()
            return fn
        hooks.append(model.model.layers[L].mlp.down_proj.register_forward_hook(make_hook(L)))

    groups = {
        'target': corpus['target'],
        'decoy_detached': corpus['decoy_detached'],
        'decoy_expressive': corpus['decoy_expressive'],
        'decoy_compliant': corpus['decoy_compliant'],
    }

    def run_group(texts, tag):
        sums = torch.zeros(n_layers, n_inter, dtype=torch.float64, device='cuda')
        for i, t in enumerate(texts):
            ids = tok(t, return_tensors='pt').to('cuda')
            with torch.no_grad():
                model(**ids)
            for L in range(n_layers):
                sums[L] += captures[L][0].float().double().mean(dim=0)
            if (i + 1) % 5 == 0:
                print(f'{tag} {i+1}/{len(texts)}', flush=True)
        captures.clear()
        torch.cuda.empty_cache()
        gc.collect()
        return (sums / len(texts)).float()

    group_means = {}
    for tag, texts in groups.items():
        print('=== group', tag, flush=True)
        group_means[tag] = run_group(texts, tag)
    for h in hooks: h.remove()

    results = {}
    for L in range(n_layers):
        t = group_means['target'][L]
        a = group_means['decoy_detached'][L]
        b = group_means['decoy_expressive'][L]
        c = group_means['decoy_compliant'][L]
        for name, d in [('detached', a), ('expressive', b), ('compliant', c)]:
            diff = t - d
            k = min(30, diff.numel())
            idx = torch.topk(diff, k=k).indices.tolist()
            vals = [round(float(diff[i]), 4) for i in idx]
            results[f'L{L:02d}_vs_{name}'] = list(zip(idx, vals))
        all_decoys = torch.stack([a, b, c]).mean(dim=0)
        diff_all = t - all_decoys
        idx = torch.topk(diff_all, k=50).indices.tolist()
        vals = [round(float(diff_all[i]), 4) for i in idx]
        results[f'L{L:02d}_vs_all'] = list(zip(idx, vals))

    with open(os.path.join(OUT_DIR, 'neuron_candidates.json'), 'w') as f:
        json.dump(results, f, indent=1)

    torch.save({tag: gm.cpu() for tag, gm in group_means.items()},
               os.path.join(OUT_DIR, 'group_means.pt'))

    best = []
    for L in range(n_layers):
        t = group_means['target'][L]
        a = group_means['decoy_detached'][L]
        b = group_means['decoy_expressive'][L]
        c = group_means['decoy_compliant'][L]
        all_decoys = torch.stack([a, b, c]).mean(dim=0)
        diff = t - all_decoys
        top = torch.topk(diff, k=10).indices.tolist()
        margins = torch.minimum(torch.minimum(t - a, t - b), t - c)
        for n in top:
            best.append({'layer': L, 'neuron': n, 'score': float(diff[n]),
                         'min_margin': float(margins[n])})
    best.sort(key=lambda x: -x['score'])
    with open(os.path.join(OUT_DIR, 'top_neurons.json'), 'w') as f:
        json.dump(best[:200], f, indent=1)

    print('--- top neurons per sampled layer (target vs all decoys) ---', flush=True)
    for L in range(0, n_layers, 4):
        t = group_means['target'][L]
        all_decoys = torch.stack([group_means['decoy_detached'][L],
                                  group_means['decoy_expressive'][L],
                                  group_means['decoy_compliant'][L]]).mean(dim=0)
        diff = t - all_decoys
        top = torch.topk(diff, k=5)
        for n, v in zip(top.indices.tolist(), top.values.tolist()):
            print(f'L{L:02d} n{n:05d} score={v:+.4f}', flush=True)

    print('DONE', flush=True)

if __name__ == '__main__':
    main()
