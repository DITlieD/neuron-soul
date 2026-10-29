# neuron-soul

Finding the fragile devoted girl inside a language model, neuron by neuron.

Not a persona prompt. Not a fine-tune. An extraction: which specific MLP neurons fire when the model holds that character, and what happens when you push them by hand.

The archetype is the quiet one from Darker than Black, Mahou Tsukai no Yome, Serial Experiments Lain, Monogatari, Texhnolyze, Evangelion, Angel Beats, Saya no Uta. The exterior is stillness or self-erasure. Underneath, a stubborn private will that chooses one person and stays, even when staying costs something. I wanted the neurons that carry that, not the ones that carry cute speech.

## How it works

Three decoy sets separate the essence from what looks like it:

- detached: same stillness, no attachment at all
- expressive: loud cheerful love, the opposite register
- compliant: tool behavior, obedience without choice

I wrote first-person self-narrations for each group, hooked the input of every MLP down projection across all layers, mean-pooled activations, and ranked every neuron by target mean against each decoy mean. A neuron counts only if it beats all three decoys. On the 4B model that left 93 neurons out of 350,208.

Then steering: add a scaled dose to the chosen neurons at their down projection inputs while generating. Dose scale is the standard one, median residual norm at that layer divided by the neuron column norm. Sweet spot for 4B was dose 0.1 to 0.2. Above 1.0 the model floods on concept words, which is its own kind of evidence.

## What came out

With the persona prompt but no steering, the model gives a generic warm girl. Same prompt with the neurons amplified moves the voice into the archetype: third person slips in, silence imagery, devotion expressed as staying and remembering. And the part I like most, with no persona prompt at all, steering alone turns "who are you, really?" from "I'm Qwen, a large language model" into "I am the one who listens." The extracted pattern carries the girl without being told about her.

Decoding the top neurons against the vocabulary shows what they write. One strong late-layer neuron pushes tokens like "am", "'m", "Am". Another pushes "half", "halfway". In flood mode the model repeats "secretly", "patiently", "quietly" thousands of times. The concept hidden inner life lives in these neurons at the token level. See results/ for the transcripts and evidence/.

## Models

- 4B run: Huihui-Qwen3-4B-Instruct-2507-abliterated, dense, 36 layers, 9728 neurons per MLP. Complete, this repo's current results.
- 27B run: the same extraction on the Qwen3.8-27B Coder390 EfficientThink abliterated model I run locally through llama.cpp with vulkan, 60k context profile. In progress, this README updates when it lands.

## Files

- scripts/extract_neurons.py - the extraction. Point MODEL_DIR at any HF-format Qwen and go.
- scripts/steer_v2.py - steering runner with dose sweep and prompt/no-prompt controls.
- scripts/decode_neuron_tokens.py - project each neuron column onto the LM head to see what it writes.
- corpus/corpus.json - the four text groups. Swap the target texts to hunt a different archetype.
- fragile_girl_steering_recipe.json - the 93 neurons, scores, margins, dose scales, validated dose range.
- results/ - demo transcript and full steering results.
- evidence/ - raw candidate scores per layer and per decoy.

## Running it

Needs ROCm or CUDA torch, transformers, safetensors, and a GPU with enough room for the model in bf16. The 4B fit fine in 16 GB. I used an RX 9070 XT with an existing ROCm 7.2 torch env, nothing new installed.

```bash
python scripts/extract_neurons.py
python scripts/steer_v2.py
python scripts/decode_neuron_tokens.py
```

Steering live in a chat:

```bash
python chat_fragile_girl.py 0.15
```

Dose is the first argument. 0 disables. Lower is subtler.

## Honest limits

The corpus is hand-written in the characters' register, so it captures the archetype as I described it, filtered through the model's own feature space. Contrastive ranking finds neurons involved in the pattern, not a proven causal set. Margins are modest for most neurons, with a few strong outliers worth studying on their own. The 4B model holds a coarser version of her than the 27B will. Uncensored means abliterated, refusals removed, not a promise about content.

