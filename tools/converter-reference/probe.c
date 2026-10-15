/* Research protocol: float32 stdin -> packed blocks + native float32 reconstruction.
 * Calls the upstream *_ref kernels directly, bypassing model and preset policy. */
#include "ggml.h"
#include "ggml-quants.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

typedef void (*encode_fn)(const float *, void *, int64_t);
typedef void (*decode_fn)(const void *, float *, int64_t);
#define CODEC(name) \
    static void encode_##name(const float * x, void * y, int64_t n) { quantize_row_##name##_ref(x, y, n); } \
    static void decode_##name(const void * x, float * y, int64_t n) { dequantize_row_##name(x, y, n); }
CODEC(q4_0)
CODEC(q4_1)
CODEC(q5_0)
CODEC(q5_1)
CODEC(q8_0)
CODEC(q2_K)
CODEC(q3_K)
CODEC(q4_K)
CODEC(q5_K)
CODEC(q6_K)

struct codec { const char * name; size_t block, bytes; encode_fn encode; decode_fn decode; };
#define ENTRY(label, name, block) { label, block, sizeof(block_##name), encode_##name, decode_##name }
static const struct codec codecs[] = {
    ENTRY("Q4_0", q4_0, 32), ENTRY("Q4_1", q4_1, 32),
    ENTRY("Q5_0", q5_0, 32), ENTRY("Q5_1", q5_1, 32), ENTRY("Q8_0", q8_0, 32),
    ENTRY("Q2_K", q2_K, 256), ENTRY("Q3_K", q3_K, 256), ENTRY("Q4_K", q4_K, 256),
    ENTRY("Q5_K", q5_K, 256), ENTRY("Q6_K", q6_K, 256)
};

static long dimension(const char * text) {
    char * end;
    errno = 0;
    long value = strtol(text, &end, 10);
    return errno || end == text || *end || value < 1 || value > 1048576 ? 0 : value;
}

int main(int argc, char ** argv) {
    if (argc != 4) { fprintf(stderr, "Expected FORMAT ROWS COLS\n"); return 2; }
    const struct codec * codec = NULL;
    for (size_t i = 0; i < sizeof(codecs) / sizeof(codecs[0]); ++i) {
        if (strcmp(argv[1], codecs[i].name) == 0) codec = &codecs[i];
    }
    long rows = dimension(argv[2]), cols = dimension(argv[3]);
    if (!codec || !rows || !cols || rows > 1048576 / cols) {
        fprintf(stderr, "Invalid format or dimensions\n"); return 2;
    }
    if ((size_t) cols % codec->block) {
        fprintf(stderr, "Incompatible row width\n"); return 2;
    }
#ifdef _WIN32
    /* Prevent Windows CR/LF and EOF translation from corrupting binary pipes. */
    if (_setmode(_fileno(stdin), _O_BINARY) < 0 || _setmode(_fileno(stdout), _O_BINARY) < 0) return 3;
#endif
    const size_t count = (size_t) rows * cols;
    const size_t row_bytes = (size_t) cols / codec->block * codec->bytes;
    float * input = malloc(count * sizeof(float));
    float * restored = malloc(count * sizeof(float));
    unsigned char * packed = malloc((size_t) rows * row_bytes);
    int result = 3;
    struct ggml_context * ctx = NULL;
    if (!input || !restored || !packed) goto done;
    if (fread(input, sizeof(float), count, stdin) != count || fgetc(stdin) != EOF) {
        fprintf(stderr, "Invalid input payload length\n"); goto done;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(input[i])) { fprintf(stderr, "Non-finite input\n"); goto done; }
    }
    /* b3962 initializes its FP16 lookup tables through ggml_init. */
    struct ggml_init_params params = {1024, NULL, true};
    ctx = ggml_init(params);
    if (!ctx) goto done;
    for (long row = 0; row < rows; ++row) {
        codec->encode(input + row * cols, packed + row * row_bytes, cols);
        codec->decode(packed + row * row_bytes, restored + row * cols, cols);
    }
    if (fwrite(packed, 1, rows * row_bytes, stdout) != rows * row_bytes ||
        fwrite(restored, sizeof(float), count, stdout) != count || fflush(stdout)) goto done;
    result = 0;
done:
    if (ctx) ggml_free(ctx);
    free(input); free(restored); free(packed);
    return result;
}
