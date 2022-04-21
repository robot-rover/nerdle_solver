#include <stdint.h>

#define DLL_EXPORT __declspec(dllexport)

typedef struct {
    // Secrets
    uint8_t *d_secret;
    size_t secret_pitch;
    size_t secret_alloc_rows;
    // Guesses
    uint8_t *d_guess;
    size_t guess_pitch;
    size_t guess_alloc_rows;
    // Clues
    uint16_t *d_clues;
    size_t clues_pitch;
} ClueContext;

#ifdef __cplusplus
extern "C" {
#endif

ClueContext DLL_EXPORT *create_context(uint32_t num_secret, uint32_t num_guess);
void DLL_EXPORT free_context(ClueContext *ctx);
int DLL_EXPORT generate_clueg(ClueContext *ctx, uint8_t *secret_eqs, uint32_t num_secret, uint8_t *guess_eqs, uint32_t num_guess, uint16_t *clue_arr);
void DLL_EXPORT helloworld();

#ifdef __cplusplus
}
#endif