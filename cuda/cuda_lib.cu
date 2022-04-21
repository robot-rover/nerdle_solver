#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cassert>
#include "cuda_lib.h"

// #define KERNEL_DEBUG
// #define API_DEBUG

#ifdef KERNEL_DEBUG
__device__ static constexpr char DEBUG_SYMBOL_TABLE[SYMBOL_ORD + 1] = "0123456789+-*/=";
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define NUM_SLOTS 8
#define SYMBOL_ORD 15

__global__ void generate_clue_kernel(uint8_t *guess, uint32_t num_guess, uint32_t guess_pitch, const uint8_t *secret, uint32_t num_secret, uint32_t secret_pitch, uint16_t *clues, uint32_t clues_pitch)
{
    __shared__ uint8_t guess_cache[32 * NUM_SLOTS];
    __shared__ uint8_t secret_cache[32 * NUM_SLOTS];
    uint8_t counts[SYMBOL_ORD];
    uint8_t clue[NUM_SLOTS];

    uint32_t guess_addr = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t guess_offset = threadIdx.x * NUM_SLOTS;
    uint32_t secret_addr = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t secret_offset = threadIdx.y * NUM_SLOTS;
    uint32_t clues_offset = guess_addr + secret_addr * (clues_pitch / sizeof(uint16_t));

    if (secret_addr < num_secret && guess_addr < num_guess)
    {
        if (threadIdx.y == 0)
        {
            uint32_t x_cache_offset = threadIdx.x * NUM_SLOTS;
            uint32_t x_main_offset = guess_addr * guess_pitch;
#ifdef KERNEL_DEBUG
            printf("Loading Guess %d (co: %d, mo: %d)\n", threadIdx.x, x_cache_offset, x_main_offset);
#endif
            for (int i = 0; i < NUM_SLOTS; i++)
            {
                guess_cache[x_cache_offset + i] = guess[x_main_offset + i];
            }
        }
        if (threadIdx.x == 0)
        {
            uint32_t y_cache_offset = threadIdx.y * NUM_SLOTS;
            uint32_t y_main_offset = secret_addr * secret_pitch;
#ifdef KERNEL_DEBUG
            printf("Loading Secret %d (co: %d, mo: %d)\n", threadIdx.y, y_cache_offset, y_main_offset);
#endif
            for (int i = 0; i < NUM_SLOTS; i++)
            {
                secret_cache[y_cache_offset + i] = secret[y_main_offset + i];
            }
        }
    }

    __syncthreads();

    if (secret_addr < num_secret && guess_addr < num_guess)
    {
#ifdef KERNEL_DEBUG
        char secret[9];
        char guess[9];
        for (int i = 0; i < NUM_SLOTS; i++) {
            secret[i] = DEBUG_SYMBOL_TABLE[secret_cache[secret_offset + i]];
            guess[i] = DEBUG_SYMBOL_TABLE[guess_cache[guess_offset + i]];
        }
        secret[8] = '\0';
        guess[8] = '\0';
        printf("(%d,%d): Guess: %s, Secret: %s\n", guess_addr, secret_addr, guess, secret);
#endif
        for (int i = 0; i < SYMBOL_ORD; i++)
        {
            counts[i] = 0;
        }
        for (int i = 0; i < NUM_SLOTS; i++)
        {
            // uint8_pitch = pitch / sizeof(uint8_t) => pitch

            uint8_t secret_symbol = secret_cache[secret_offset + i];
            uint8_t guess_symbol = guess_cache[guess_offset + i];
            uint8_t zero_if_green = secret_symbol == guess_symbol ? 0 : 1;
            counts[secret_symbol] += zero_if_green;
            clue[i] = zero_if_green;
#ifdef KERNEL_DEBUG
            printf("(%d,%d): A%d SecretSym: %d, GuessSym: %d, ZIG: %d\n", guess_addr, secret_addr, i, secret_symbol, guess_symbol, zero_if_green);
#endif
        }
#ifdef KERNEL_DEBUG
        char count_str[SYMBOL_ORD*4 + 1];
        count_str[SYMBOL_ORD*4] = '\0';
        for (int i = 0; i < SYMBOL_ORD; i++) {
            count_str[i*4] = DEBUG_SYMBOL_TABLE[i];
            count_str[i*4+1] = ':';
            count_str[i*4+2] = '0' + counts[i];
            count_str[i*4+3] = ' ';
        }
        printf("(%d,%d): Counts: %s\n", guess_addr, secret_addr, count_str);
#endif
        for (int i = 0; i < NUM_SLOTS; i++)
        {
            uint8_t guess_symbol = guess_cache[guess_offset + i];
            uint8_t one_if_yellow = clue[i] & (counts[guess_symbol] > 0 ? 1 : 0);
            counts[guess_symbol] -= one_if_yellow;
            clue[i] += one_if_yellow;
#ifdef KERNEL_DEBUG
            printf("(%d,%d): B%d GuessSym: %d, OIY: %d\n", guess_addr, secret_addr, i, guess_symbol, one_if_yellow);
#endif
        }
#ifdef KERNEL_DEBUG
        printf("(%d,%d): Clue Offset: %d\n", guess_addr, secret_addr, clues_offset);
#endif
        uint16_t clue_packed = 0;
        for (int i = 0; i < NUM_SLOTS; i++)
        {
            clue_packed *= 3;
            clue_packed += clue[i];
        }
        clues[clues_offset] = clue_packed;
    }
}

extern "C" ClueContext* create_context(uint32_t num_guess, uint32_t num_secret) {
    ClueContext *ctx = (ClueContext*)malloc(sizeof(ClueContext));
    ctx->secret_alloc_rows = num_secret;
    ctx->guess_alloc_rows = num_guess;

    // Size of host eqs (and retval clues)
    size_t eq_width = sizeof(uint8_t) * NUM_SLOTS;

    // Create guess on device (2D)
    gpuErrchk(cudaMallocPitch((void **)&ctx->d_guess, &ctx->guess_pitch, eq_width, num_guess));

    // Create secret on device (2D)
    gpuErrchk(cudaMallocPitch((void **)&ctx->d_secret, &ctx->secret_pitch, eq_width, num_secret));

    // Create clues on device (2D)
    gpuErrchk(cudaMallocPitch((void **)&ctx->d_clues, &ctx->clues_pitch, sizeof(uint16_t) * num_guess, num_secret));

    gpuErrchk(cudaDeviceSynchronize());

    return ctx;
}

extern "C" void free_context(ClueContext *ctx) {
    // Free device global memory
    gpuErrchk(cudaFree(ctx->d_secret));
    gpuErrchk(cudaFree(ctx->d_guess));
    gpuErrchk(cudaFree(ctx->d_clues));
    gpuErrchk(cudaDeviceSynchronize());
    free(ctx);
}

extern "C" int generate_clueg(ClueContext *ctx, uint8_t *guess_eqs, uint32_t num_guess, uint8_t *secret_eqs, uint32_t num_secret, uint16_t *clue_arr)
{
    size_t eq_width = sizeof(uint8_t) * NUM_SLOTS;
    if (num_guess > ctx->guess_alloc_rows) {
        return -1;
    }
    if (num_secret > ctx->secret_alloc_rows) {
        return -2;
    }

    // Copy to the device
    gpuErrchk(cudaMemcpy2D(ctx->d_guess, ctx->guess_pitch, guess_eqs, eq_width, eq_width, num_guess, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy2D(ctx->d_secret, ctx->secret_pitch, secret_eqs, eq_width, eq_width, num_secret, cudaMemcpyHostToDevice));

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 32;
    int blocksPerGridx = (num_guess + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridy = (num_secret + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blocks(blocksPerGridx, blocksPerGridy);
    dim3 threads(threadsPerBlock, threadsPerBlock);

#ifdef API_DEBUG
    printf("CUDA: kernel launch with %dx%d blocks of %d threads\n", blocksPerGridx, blocksPerGridy, threadsPerBlock);
#endif
    generate_clue_kernel<<<blocks, threads>>>(ctx->d_guess, num_guess, ctx->guess_pitch, ctx->d_secret, num_secret, ctx->secret_pitch, ctx->d_clues, ctx->clues_pitch);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    gpuErrchk(cudaMemcpy2D(clue_arr, sizeof(uint16_t)*num_guess, ctx->d_clues, ctx->clues_pitch, sizeof(uint16_t) * num_guess, num_secret, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());

    return 0;
}

extern "C" void helloworld() {
    printf("Hello World\n");
}