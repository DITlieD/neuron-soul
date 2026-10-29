#!/usr/bin/env python3
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = '/home/lied/Documents/Codex/2026-10-29/read/work'
MODEL_DIR = WORK + '/model'
RECIPE = WORK + '/../outputs/fragile_girl_steering_recipe.json'

NEUTRAL = [
    "The weather today is mild with light winds from the southwest.",
    "To prepare the dough, mix flour, water, salt, and yeast, then knead for ten minutes.",
    "The train arrives at platform four at half past six every morning.",
    "Compile the report, attach the appendix, and send it to the committee.",
]

def main():
    with open(RECIPE) as f:
        recipe = json.load(f)
    by_layer = {}
    for x in recipe['neurons']:
        by_layer.setdefault(x['layer'], []).append((x['neuron'], x['dose_scale']))

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa')
    model.eval()
    n_layers = model.config.num_hidden_layers

    resid_norms = torch.zeros(n_layers, dtype=torch.float32)
    cap = {}
    hooks = []
    for L in range(n_layers):
        def mk(L):
            def fn(module, args, output):
                cap[L] = args[0].detach()
            return fn
        hooks.append(model.model.layers[L].mlp.down_proj.register_forward_hook(mk(L)))
    for t in NEUTRAL:
        ids = tok(t, return_tensors='pt').to('cuda')
        with torch.no_grad():
            model(**ids)
        for L in range(n_layers):
            resid_norms[L] += cap[L][0].float().norm(dim=-1).median().cpu()
    for L in range(n_layers):
        resid_norms[L] /= len(NEUTRAL)
    for h in hooks: h.remove()
    cap.clear()

    active = {'dose': float(sys.argv[1]) if len(sys.argv) > 1 else 0.15}
    hooks2 = []
    for L, neurons in by_layer.items():
        def mk_pre(L, neurons):
            def fn(module, args):
                if active['dose'] <= 0:
                    return None
                out = args[0].clone()
                for n, base_scale in neurons:
                    live_scale = (resid_norms[L] / (base_scale * model.model.layers[L].mlp.down_proj.weight[:, n].float().norm())).item()
                    out[..., n] += active['dose'] * live_scale
                return (out,)
            return fn
        hooks2.append(model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk_pre(L, neurons)))

    SYSTEM = recipe['system_prompt']
    print(f"fragile-girl chat, dose={active['dose']} (try 0.05-0.3; 0 disables). Ctrl+C to quit.")
    history = [{'role': 'system', 'content': SYSTEM}]
    while True:
        try:
            user = input('you> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        history.append({'role': 'user', 'content': user})
        text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=250, do_sample=True, temperature=0.8,
                                 top_p=0.9, repetition_penalty=1.15, pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        history.append({'role': 'assistant', 'content': resp})
        print(f'girl> {resp}')
        print()

if __name__ == '__main__':
    main()
