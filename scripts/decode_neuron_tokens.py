#!/usr/bin/env python3
import json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = '/home/lied/Documents/Codex/2026-10-29/read/work'
MODEL_DIR = os.path.join(WORK, 'model')
OUT = os.path.join(WORK, 'evidence', 'neuron_token_decoding.json')

def main():
    with open(os.path.join(WORK, 'evidence', 'top_neurons.json')) as f:
        tops = json.load(f)
    chosen = tops[:40]
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa')
    model.eval()
    lm_w = model.get_output_embeddings().weight
    results = []
    for x in chosen:
        L, n = x['layer'], x['neuron']
        col = model.model.layers[L].mlp.down_proj.weight[:, n].to(lm_w.dtype)
        scores = col.to('cuda') @ lm_w.to('cuda').T
        top = torch.topk(scores, k=8)
        bot = torch.topk(-scores, k=4)
        def toks(idx):
            return [tok.decode([int(i)]).strip() or tok.convert_ids_to_tokens([int(i)])[0] for i in idx]
        results.append({'layer': L, 'neuron': n, 'score': round(x['score'], 3), 'min_margin': round(x['min_margin'], 3),
                        'writes': toks(top.indices), 'suppresses': toks(bot.indices)})
        print('L%02d n%05d writes: %s' % (L, n, results[-1]['writes']), flush=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
