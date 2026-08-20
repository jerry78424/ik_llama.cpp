// DSpark Markov + confidence head.
// Declared here (not on llm_build_context) so that signature changes only
// recompile the graph TUs that actually call it, not every includer of
// llama-build-context.h.
#pragma once

struct ggml_tensor;
struct llm_build_context;

// Applies the DSpark Markov bias chain to `base_logits` in a block-parallel
// strided fashion (one chain per unique sequence in the batch), returns the
// biased [n_vocab, n_tokens] logits, writes the greedy argmax draft tokens
// (block-major, aligned with the returned logits) to *draft_tokens_out, and,
// when the model has a confidence projection and `input_embd` is provided,
// writes the per-position acceptance probabilities [1, n_tokens] named
// "dspark_conf" to *conf_out.
ggml_tensor * build_dspark_logits(
        llm_build_context & llm,
        ggml_tensor * base_logits,
        ggml_tensor * input_tokens,
        ggml_tensor * input_embd,
        ggml_tensor ** draft_tokens_out = nullptr,
        ggml_tensor ** conf_out = nullptr);
