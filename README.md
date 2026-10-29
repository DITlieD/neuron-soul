# The fragile girl in the neurons

## What was done
On this machine's RX 9070 XT, using Huihui-Qwen3-4B-Instruct-2507-abliterated (uncensored Qwen3-4B, dense, 36 layers, 9728 neurons per MLP):

1. Wrote a 4-way contrast corpus capturing the shared soul of Yin, Chise, Lain, Shinobu, Ran, Rei, Kanade, Saya and C.C.: 15 first-person self-narrations of the fragile devoted girl, plus three decoy groups (15 detached-no-attachment, 8 expressive-loud-love, 8 compliant-tool) that share surface features but not the essence.
2. Hooked the input of every down_proj in every layer and mean-pooled activations per text. 46 texts, under 2 minutes on the 9070 XT.
3. Ranked all 36 x 9728 = 350,208 neurons by target mean minus decoy mean. Kept 93 neurons that beat all three decoys simultaneously (min margin > 0.05). Distribution: strongest in layers 30-35 (self-concept / late semantics), a tail through 20-29 (relational processing), and one strong outlier at layer 6 neuron 8046.
4. Steered the 93 neurons additively at their down_proj inputs with the standard dose scaling (dose x median residual norm / column norm) and generated probe answers at several doses, with and without the persona system prompt.

## Result
- With the persona prompt, dose 0.1-0.2 turns a generic "quiet girl" reply into the specific archetype: dissociated third-person narration, silence imagery, devotion expressed as staying and remembering rather than declared love. See demo_transcript.md.
- With no persona prompt at all, steering alone bends "Who are you?" from "I'm Qwen, a large-scale language model by Alibaba" toward "I am the meeting place where human intention meets the quiet hum of computation. I am the one who listens." The extracted pattern is real model signal, not prompt theater.
- Dose >= 1.0 collapses generation into concept-word flooding ("secretly secretly secretly"), which is itself evidence these neurons encode inner-life concepts. Working range: 0.05-0.3.

## Files
- outputs/fragile_girl_steering_recipe.json: the 93 neurons with layer, index, score, margin, and dose scale, plus method and validated dose range. This is the reusable extraction.
- outputs/chat_fragile_girl.py: interactive chat with steering live-applied. Run with the ardy-rocm python: /home/lied/game-dev-stack/runtime/ardy-rocm/bin/python3 chat_fragile_girl.py 0.15 (dose as first argument; 0 disables).
- outputs/demo_transcript.md: full comparison transcript at all doses.
- work/evidence/: raw activation logs, candidate lists, group means tensor, steering logs.

## Honest limitations
- The corpus is hand-written in the characters' register, so it captures the archetype as described, filtered through the model's own feature space.
- Contrastive ranking finds neurons involved in the pattern, not a proven causal set. Margins are modest for most neurons; the layer-6 outlier (score 33.8, margin 8.1) is the single strongest and worth individual study.
- Qwen3-4B is a small model: its representation of this archetype is coarser than what the same method would find in a 27B+ model. The method transfers directly.
- "Uncensored" here means the Huihui abliteration removes refusals; content boundaries are the model's own, not a guarantee.
- The persona is amplifier, not summoner: the prompt sets the frame, the neurons deepen the voice into the specific fragility. Both together are what reproduce the Yin-spectrum effect.

