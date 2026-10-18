/* Research-only bounded C ABI over preserved MIT-licensed b3962 reference math.
 * No model policy, allocation ownership transfer, file I/O, or implicit fallback.
 * Upstream license: third_party/converter_references/llama-b3962-LICENSE. */
#include "ggml.h"
#include "ggml-quants.h"
#include <math.h>
#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* Return 1 for invalid arguments, 2 for nonfinite input, 3 for init failure.
 * All validation precedes output writes. The caller owns both buffers, keeps
 * input immutable, and must provide nonoverlapping input/output allocations. */
EXPORT int converter_encode(int bits, const float * input, size_t count,
                            uint8_t * output, size_t capacity) {
    size_t bytes = bits == 2 ? 84 : bits == 3 ? 110 : bits == 4 ? 144 : bits == 5 ? 176 : bits == 6 ? 210 : 0;
    if (!bytes || !input || !output || !count || count % 256 || count > 1048576 ||
        capacity != count / 256 * bytes) return 1;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(input[i])) return 2;
    }
    /* b3962 requires initialized FP16 lookup tables, even for raw row kernels. */
    struct ggml_init_params params = {1024, NULL, true};
    struct ggml_context * context = ggml_init(params);
    if (!context) return 3;
    if (bits == 2) quantize_row_q2_K_ref(input, (block_q2_K *) output, (int64_t) count);
    if (bits == 3) quantize_row_q3_K_ref(input, (block_q3_K *) output, (int64_t) count);
    if (bits == 4) quantize_row_q4_K_ref(input, (block_q4_K *) output, (int64_t) count);
    if (bits == 5) quantize_row_q5_K_ref(input, (block_q5_K *) output, (int64_t) count);
    if (bits == 6) quantize_row_q6_K_ref(input, (block_q6_K *) output, (int64_t) count);
    ggml_free(context);
    return 0;
}
