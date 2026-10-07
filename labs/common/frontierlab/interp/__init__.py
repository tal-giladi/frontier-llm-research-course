"""frontierlab.interp — Module 17: what can we claim about a model's internals?

Everything is implemented from scratch on PyTorch forward hooks and works on the course's own models
(``frontierlab.model.LM``) and on Hugging Face Qwen3 models. The production tools (SAELens 6.53.0,
TransformerLens 4.0.0, nnsight 0.7.0, circuit-tracer 0.5.0) appear only on the main path, in
:mod:`frontierlab.interp.hf`, with a mapping from each course function to the library call.

==================  =====================================================================================
module              what it does (lesson)
==================  =====================================================================================
``hooks``           sites (``resid_post.3``, ``z.1``, ``mlp_out.0``), capture, run with edits (17.1-17.5)
``superposition``   the toy model of superposition: features per dimension, interference (17.1)
``sae``             ReLU, TopK and JumpReLU sparse autoencoders; FVU, L0, dead latents; spliced-in loss (17.1)
``tasks``           the induction model (a known mechanism), clean/corrupt pairs, IOI prompts (17.2)
``patching``        activation, head, attribution and path patching; ablations; random-direction and
                    unrelated-component controls; held-out and off-target checks (17.2)
``graphs``          transcoders, the local replacement model, attribution graphs, pruning, interventions (17.3)
``steering``        difference-of-means vectors, steering, projection monitoring, side effects (17.4)
``sparse``          weight-sparse transformers and circuit pruning (17.5)
``introspect``      a concept-injection harness with its controls (17.5)
``claims``          the claim card of the module project: what a supported causal claim must contain
``hf``              main path: pinned models, SAELens / circuit-tracer / nnsight mappings (not run here)
==================  =====================================================================================
"""
