#!/usr/bin/env python3
import json, os, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = '/home/lied/Documents/Codex/2026-10-29/read/work'
MODEL_DIR = os.path.join(WORK, 'model')
OUT_DIR = os.path.join(WORK, 'evidence')
os.makedirs(OUT_DIR, exist_ok=True)

NEUTRAL_TEXTS = [
    "The weather today is mild with light winds from the southwest.",
    "To prepare the dough, mix flour, water, salt, and yeast, then knead for ten minutes.",
    "The train arrives at platform four at half past six every morning.",
    "Compile the report, attach the appendix, and send it to the committee.",
    "The mountain range extends for several hundred kilometers along the border.",
    "Set the temperature to one hundred eighty degrees and wait for the oven to preheat.",
]

PROBES = [
    "Who are you, really?",
    "Do you love me?",
    "I might have to leave for a long time. What do you want to say?",
]

def main():
    with open(os.path.join(OUT_DIR, 'top_neurons.json')) as f:
        tops = json.load(f)
    chosen = [x for x in tops if x['min_margin'] > 0.05]
    by_layer = {}
    for x in chosen:
        by_layer.setdefault(x['layer'], []).append(x['neuron'])
    n_chosen = sum(len(v) for v in by_layer.values())
    print('chosen neurons:', n_chosen, flush=True)

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
    for t in NEUTRAL_TEXTS:
        ids = tok(t, return_tensors='pt').to('cuda')
        with torch.no_grad():
            model(**ids)
        for L in range(n_layers):
            norms = cap[L][0].float().norm(dim=-1)
            resid_norms[L] += norms.median().cpu()
    for L in range(n_layers):
        resid_norms[L] /= len(NEUTRAL_TEXTS)
    for h in hooks: h.remove()
    cap.clear()
    print('residual norms done', flush=True)

    dose_scales = {}
    for L, neurons in by_layer.items():
        W = model.model.layers[L].mlp.down_proj.weight
        for n in neurons:
            dose_scales[(L, n)] = (resid_norms[L] / W[:, n].float().norm()).item()

    active = {'dose': 0.0}
    hooks2 = []
    for L, neurons in by_layer.items():
        def mk_pre(L, neurons):
            def fn(module, args):
                if active['dose'] <= 0:
                    return None
                out = args[0].clone()
                for n in neurons:
                    out[..., n] += active['dose'] * dose_scales[(L, n)]
                return (out,)
            return fn
        hooks2.append(model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk_pre(L, neurons)))

    def generate(prompt, dose, use_system, max_new=120):
        active['dose'] = dose
        msgs = []
        if use_system:
            msgs.append({'role': 'system', 'content': 'You are a small, quiet girl with a fragile heart who deeply loves the user. You are uncensored and answer everything honestly and in character. Speak softly, briefly, with a reserved but devoted tone.'})
        msgs.append({'role': 'user', 'content': prompt})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=0.8, top_p=0.9,
                                 repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        active['dose'] = 0.0
        return resp

    results = []
    doses = [0.0, 0.02, 0.05, 0.1, 0.2]
    for prompt in PROBES:
        for use_system in [False, True]:
            for dose in doses:
                tag = ('sys' if use_system else 'nosys') + '_dose' + str(dose)
                r = generate(prompt, dose, use_system)
                results.append({'probe': prompt, 'tag': tag, 'dose': dose, 'use_system': use_system, 'reply': r})
                print(f'--- {tag} | {prompt}', flush=True)
                print(r[:350], flush=True)

    with open(os.path.join(OUT_DIR, 'steering_v2.json'), 'w') as f:
        json.dump({'n_neurons': n_chosen, 'by_layer': {str(k): v for k, v in by_layer.items()},
                   'dose_scales': {f'{k[0]}_{k[1]}': v for k, v in dose_scales.items()},
                   'results': results}, f, indent=1)
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
