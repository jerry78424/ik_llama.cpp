#include "common.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// Deterministic greedy decode that dumps the FULL logits vector for every
// sampled position to a binary file, so two builds can be compared logit by
// logit (byte-identical file == identical outputs). Used by the build.bat
// logit gate (tools/build-gate.ps1) to catch silent output changes from a
// rebuild before they reach production.
//
// Output format (little-endian, binary):
//   int32 magic   = 0x4C47544C ("LGTL")
//   int32 n_vocab
//   per decoded step:
//     int32 token_id          (the greedy-chosen token for that step)
//     n_vocab float32 logits  (the full logits used to sample it)
// The prompt's final position is the first step; each generated token follows.

static void print_usage(int argc, char ** argv, const gpt_params & params) {
    gpt_params_print_usage(argc, argv, params);
    LOG_TEE("\nexample usage:\n");
    LOG_TEE("\n    %s -m model.gguf -f prompt.txt -n 32 --temp 0 --seed 0 --logit-dump dump.bin\n", argv[0]);
    LOG_TEE("\n");
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.n_predict = 16;

    std::string logit_dump_file;

    // Pull the custom --logit-dump out of argv BEFORE gpt_params_parse: this
    // fork's parser rejects unknown arguments.
    std::vector<const char *> filtered;
    filtered.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--logit-dump" && i + 1 < argc) {
            logit_dump_file = argv[++i];
            continue;
        }
        filtered.push_back(argv[i]);
    }
    const int argc2 = (int) filtered.size();
    char ** argv2 = (char **) filtered.data();

    if (!gpt_params_parse(argc2, argv2, params)) {
        print_usage(argc, argv, params);
        return 1;
    }
    if (logit_dump_file.empty()) {
        fprintf(stderr, "%s: --logit-dump <file> is required\n", __func__);
        print_usage(argc, argv, params);
        return 1;
    }

    // Force deterministic greedy decoding: no RNG, no temperature.
    params.sparams.temp = 0.0f;
    params.sparams.seed = 0;

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    std::string prompt = params.prompt;
    if (!params.prompt_file.empty()) {
        std::ifstream pf(params.prompt_file, std::ios::binary);
        if (pf) {
            prompt.assign((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
        } else {
            fprintf(stderr, "%s: error: cannot read prompt file %s\n", __func__, params.prompt_file.c_str());
            return 1;
        }
    }
    if (prompt.empty()) {
        fprintf(stderr, "%s: error: empty prompt\n", __func__);
        return 1;
    }

    const std::vector<llama_token> tokens_list = common_tokenize(ctx, prompt, true);
    const int n_vocab = llama_n_vocab(model);
    const int n_predict = std::max(1, params.n_predict);

    LOG_TEE("%s: prompt %zu tokens, n_predict %d, n_vocab %d, n_ctx %d\n",
            __func__, tokens_list.size(), n_predict, n_vocab, llama_n_ctx(ctx));

    std::ofstream out(logit_dump_file, std::ios::binary | std::ios::trunc);
    if (!out) {
        fprintf(stderr, "%s: error: cannot open output file %s\n", __func__, logit_dump_file.c_str());
        return 1;
    }
    const int32_t magic = 0x4C47544C;
    out.write((const char *) &magic, sizeof(magic));
    out.write((const char *) &n_vocab, sizeof(n_vocab));

    // Sample the greedy token from the given logits row and dump that row.
    const auto sample_and_dump = [&](const float * logits) -> int {
        if (logits == nullptr) {
            fprintf(stderr, "%s: error: no logits available\n", __func__);
            return -1;
        }
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

        out.write((const char *) &new_token_id, sizeof(new_token_id));
        out.write((const char *) logits, (std::streamsize) n_vocab * sizeof(float));
        return (int) new_token_id;
    };

    const int n_ubatch = std::max(1, (int) llama_n_ubatch(ctx));
    llama_batch batch = llama_batch_init(n_ubatch, 0, 1);

    // Evaluate the prompt in chunks; only the final prompt position needs logits.
    for (size_t off = 0; off < tokens_list.size(); off += (size_t) n_ubatch) {
        common_batch_clear(batch);
        const size_t n = std::min<size_t>((size_t) n_ubatch, tokens_list.size() - off);
        for (size_t i = 0; i < n; ++i) {
            common_batch_add(batch, tokens_list[off + i], (llama_pos) (off + i), { 0 }, false);
        }
        batch.logits[batch.n_tokens - 1] = true;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "%s: error: prompt decode failed\n", __func__);
            return 1;
        }
    }

    const int n_decode_max = n_predict + 1;
    int n_decode = 0;
    llama_token last = -1;

    // First step: the prompt's final position.
    const float * prompt_logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    last = (llama_token) sample_and_dump(prompt_logits);
    if (last < 0) return 1;
    n_decode += 1;

    while (!llama_token_is_eog(model, last) && n_decode < n_decode_max) {
        common_batch_clear(batch);
        common_batch_add(batch, last, (llama_pos) (tokens_list.size() + n_decode - 1), { 0 }, true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "%s: error: decode failed\n", __func__);
            return 1;
        }
        last = (llama_token) sample_and_dump(llama_get_logits_ith(ctx, 0));
        if (last < 0) return 1;
        n_decode += 1;
    }

    out.close();
    LOG_TEE("%s: wrote %d logit records to %s\n", __func__, n_decode, logit_dump_file.c_str());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
