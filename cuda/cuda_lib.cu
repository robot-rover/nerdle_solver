#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cub/cub.cuh>
#include "cuda_lib.h"

// #define CLUE_DEBUG
// #define COUNTS_DEBUG
// #define API_DEBUG

#define NUM_SLOTS 8
#define SYMBOL_ORD 15
#define COUNT_SLOTS 200


#ifdef CLUE_DEBUG
__device__ static constexpr char SYMBOL_TABLE[SYMBOL_ORD + 1] = "0123456789+-*/=";
__device__ static constexpr char CLUE_TABLE[3 + 1] = "gby";
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

__global__ void sorted_counts_kernel(uint16_t *clues, uint32_t clue_pitch, uint32_t num_guess, uint32_t num_secret, double* entropies) {
    uint32_t guess_idx = blockIdx.x;
    uint32_t counted = 1;
    uint16_t *clue_offset = (uint16_t*)(((uint8_t*)clues) + guess_idx * clue_pitch);
    double entropy = 0;

    for (uint32_t i = 1; i < num_secret; i++) {
#ifdef COUNTS_DEBUG
        printf("(%d): Clue ID: %d\n", guess_idx, clue_offset[i-1]);
#endif
        if (clue_offset[i] == clue_offset[i-1]) {
            counted += 1;
        } else {
            double probability = ((double)counted) / ((double)num_secret);
            entropy -= probability * log2(probability);
#ifdef COUNTS_DEBUG
            printf("(%d): Prob for clue %d: %d/%d = %f\n", guess_idx, clue_offset[i-1], counted, num_secret, probability);
#endif
            counted = 1;
        }
    }
    double probability = ((double)counted) / ((double)num_secret);
    entropy -= probability * log2(probability);
#ifdef COUNTS_DEBUG

    printf("(%d): Clue ID: %d\n", guess_idx, clue_offset[num_secret-1]);
    printf("(%d): Prob for clue %d: %d/%d = %f\n", guess_idx, clue_offset[num_secret-1], counted, num_secret, probability);
#endif

#ifdef COUNTS_DEBUG
    printf("(%d): Result: %f\n", guess_idx, entropy);
#endif
    entropies[guess_idx] = entropy;
}

typedef struct {
    uint16_t clue;
    uint16_t count;
} ClueCount;

__global__ void clue_counts_kernel(uint16_t *clues, uint32_t clue_pitch, uint32_t num_guess, uint32_t num_secret, double* entropies) {
    uint32_t guess_idx = blockIdx.x;
    ClueCount counts[COUNT_SLOTS];
    uint32_t num_counts = 0;

    for (uint32_t secret_index = 0; secret_index < num_secret; secret_index++) {
        bool found = false;
        uint32_t clue_index = secret_index + guess_idx * (clue_pitch / sizeof(uint16_t));
        uint16_t clue = clues[clue_index];
        uint32_t count_index;
        for (count_index = 0; count_index < num_counts; count_index++) {
            if (counts[count_index].clue == clue) {
                counts[count_index].count += 1;
                found = true;
#ifdef COUNTS_DEBUG
                printf("(%d): Found Bucket for %d, new count: %d\n", guess_idx, clue, counts[count_index].count);
#endif
                break;
            }
        }
        if (found) {
            uint16_t temp_count;
            for(; count_index > 0; count_index--) {
                if (counts[count_index].count > counts[count_index-1].count) {
#ifdef COUNTS_DEBUG
                printf("(%d): Shuffling %d|#%d <=> %d|#%d\n", guess_idx, counts[count_index-1].clue, counts[count_index-1].count, clue, counts[count_index].count);
#endif
                    counts[count_index].clue = counts[count_index-1].clue;
                    counts[count_index-1].clue = clue;

                    temp_count = counts[count_index].count;
                    counts[count_index].count = counts[count_index-1].count;
                    counts[count_index-1].count = temp_count;
                } else {
                    break;
                }
            }
        } else {
#ifdef COUNTS_DEBUG
                printf("(%d): Creating New for %d @ index %d\n", guess_idx, clue, num_counts);
#endif
            if (num_counts == COUNT_SLOTS) {
#ifdef COUNTS_DEBUG
                printf("Slots full, returning error\n");
#endif
                entropies[guess_idx] = NAN;
                return;
            }
            counts[num_counts].clue = clue;
            counts[num_counts].count = 1;
            num_counts += 1;
        }
    }

    double entropy = 0;
    for (int count_index = 0; count_index < num_counts; count_index++) {
        double probability = ((double)counts[count_index].count) / ((double)num_secret);
#ifdef COUNTS_DEBUG
        printf("(%d): Prob for clue %d: %d/%d = %f\n", guess_idx, counts[count_index].clue, counts[count_index].count, num_secret, probability);
#endif
        entropy -= probability * log2(probability);
    }

#ifdef COUNTS_DEBUG
    printf("(%d): Result: %f\n", guess_idx, entropy);
#endif
    entropies[guess_idx] = entropy;
}

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
    uint32_t clues_offset = secret_addr + guess_addr * (clues_pitch / sizeof(uint16_t));

    if (secret_addr < num_secret && guess_addr < num_guess)
    {
        if (threadIdx.y == 0)
        {
            uint32_t x_cache_offset = threadIdx.x * NUM_SLOTS;
            uint32_t x_main_offset = guess_addr * guess_pitch;
#ifdef CLUE_DEBUG
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
#ifdef CLUE_DEBUG
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
#ifdef CLUE_DEBUG
        char secret[9];
        char guess[9];
        for (int i = 0; i < NUM_SLOTS; i++) {
            secret[i] = SYMBOL_TABLE[secret_cache[secret_offset + i]];
            guess[i] = SYMBOL_TABLE[guess_cache[guess_offset + i]];
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
#ifdef CLUE_DEBUG
            printf("(%d,%d): A%d SecretSym: %d, GuessSym: %d, ZIG: %d\n", guess_addr, secret_addr, i, secret_symbol, guess_symbol, zero_if_green);
#endif
        }
#ifdef CLUE_DEBUG
        char count_str[SYMBOL_ORD*4 + 1];
        count_str[SYMBOL_ORD*4] = '\0';
        for (int i = 0; i < SYMBOL_ORD; i++) {
            count_str[i*4] = SYMBOL_TABLE[i];
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
#ifdef CLUE_DEBUG
            printf("(%d,%d): B%d GuessSym: %d, OIY: %d\n", guess_addr, secret_addr, i, guess_symbol, one_if_yellow);
#endif
        }
        uint16_t clue_packed = 0;
        for (int i = 0; i < NUM_SLOTS; i++)
        {
            clue_packed *= 3;
            clue_packed += clue[i];
        }
#ifdef CLUE_DEBUG
        char clue_str[NUM_SLOTS + 1];
        clue_str[NUM_SLOTS] = '\0';
        for (int i = 0; i < NUM_SLOTS; i++) {
            clue_str[i] = CLUE_TABLE[clue[i]];
        }
        printf("(%d,%d): Clue: %s, Clue Offset: %d, Packed: %d\n", guess_addr, secret_addr, clue_str, clues_offset, clue_packed);
#endif
        clues[clues_offset] = clue_packed;
    }
}

extern "C" ClueContext* create_context(uint32_t num_guess, uint32_t num_secret) {
    ClueContext *ctx = (ClueContext*)malloc(sizeof(ClueContext));
    ctx->secret_alloc_rows = num_secret;
    ctx->guess_alloc_rows = num_guess;

    // Size of host eqs
    size_t eq_width = sizeof(uint8_t) * NUM_SLOTS;

    size_t dont_care;
    gpuErrchk(cudaMallocPitch((void **)&ctx->d_guess, &ctx->guess_pitch, eq_width, num_guess));
    gpuErrchk(cudaMallocPitch((void **)&ctx->d_secret, &ctx->secret_pitch, eq_width, num_secret));
    gpuErrchk(cudaMallocPitch((void **)&ctx->d_clues, &ctx->clues_pitch, sizeof(uint16_t) * num_secret, num_guess));
    gpuErrchk(cudaMallocPitch((void **)&ctx->d_clues_alt, &dont_care, sizeof(uint16_t) * num_secret, num_guess));
    assert(dont_care == ctx->clues_pitch);

    gpuErrchk(cudaMalloc((void **)&ctx->d_entropies, sizeof(double) * num_guess));

    uint32_t *scratch_begin = (uint32_t*)malloc(num_guess * sizeof(uint32_t));
    uint32_t *scratch_end = (uint32_t*)malloc(num_guess * sizeof(uint32_t));
    for (uint32_t i = 0; i <= num_guess; i++) {
        uint32_t begin_offset = i * (ctx->clues_pitch / sizeof(uint16_t));
        scratch_begin[i] = begin_offset;
        scratch_end[i] = begin_offset + num_secret;
#ifdef API_DEBUG
        printf("API: segment %d in indexes [%d,%d)\n", i, begin_offset, begin_offset + num_secret);
#endif
    }
    gpuErrchk(cudaMalloc((void **)&ctx->d_clue_begin, sizeof(uint32_t) * num_guess));
    gpuErrchk(cudaMalloc((void **)&ctx->d_clue_end, sizeof(uint32_t) * num_guess));
    gpuErrchk(cudaMemcpy(ctx->d_clue_begin, scratch_begin, sizeof(uint32_t) * num_guess, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(ctx->d_clue_end, scratch_end, sizeof(uint32_t) * num_guess, cudaMemcpyHostToDevice));
    free(scratch_begin);
    free(scratch_end);

    gpuErrchk(cudaDeviceSynchronize());

    return ctx;
}

extern "C" void free_context(ClueContext *ctx) {
    // Free device global memory
    gpuErrchk(cudaFree(ctx->d_secret));
    gpuErrchk(cudaFree(ctx->d_guess));
    gpuErrchk(cudaFree(ctx->d_clues));
    gpuErrchk(cudaFree(ctx->d_clues_alt));
    gpuErrchk(cudaFree(ctx->d_entropies));
    gpuErrchk(cudaFree(ctx->d_clue_begin));
    gpuErrchk(cudaFree(ctx->d_clue_end));

    gpuErrchk(cudaDeviceSynchronize());

    free(ctx);
}

extern "C" int generate_entropies(ClueContext *ctx, uint32_t num_guess, uint32_t num_secret, double *entropies, bool use_sort_alg) {
    if (num_guess > ctx->guess_alloc_rows) {
        return -1;
    }
    if (num_secret > ctx->secret_alloc_rows) {
        return -2;
    }
    if (num_guess < 1) {
        return -5;
    }
    if (num_secret < 1) {
        return -6;
    }

#ifdef API_DEBUG
    printf("API: kernel launch with %d blocks of %d threads\n", num_guess, 1);
#endif
    if (use_sort_alg) {
        cub::DoubleBuffer<uint16_t> d_dbuf(ctx->d_clues, ctx->d_clues_alt);

        void *d_temp_storage = NULL;
        size_t temp_storage_size = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_size, d_dbuf, num_secret*num_guess, num_guess, ctx->d_clue_begin, ctx->d_clue_end, 0, 13);
        gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_size));
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_size, d_dbuf, num_secret*num_guess, num_guess, ctx->d_clue_begin, ctx->d_clue_end, 0, 13);
        sorted_counts_kernel<<<num_guess, 1>>>(d_dbuf.Current(), ctx->clues_pitch, num_guess, num_secret, ctx->d_entropies);
    } else {
        clue_counts_kernel<<<num_guess, 1>>>(ctx->d_clues, ctx->clues_pitch, num_guess, num_secret, ctx->d_entropies);
    }

    gpuErrchk(cudaMemcpy(entropies, ctx->d_entropies, num_guess * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());
    return 0;
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
    if (guess_eqs == NULL) {
        return -3;
    }
    if (secret_eqs == NULL) {
        return -4;
    }
    if (num_guess < 1) {
        return -5;
    }
    if (num_secret < 1) {
        return -6;
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
    if (clue_arr != NULL) {
        gpuErrchk(cudaMemcpy2D(clue_arr, sizeof(uint16_t)*num_secret, ctx->d_clues, ctx->clues_pitch, sizeof(uint16_t) * num_secret, num_guess, cudaMemcpyDeviceToHost));
    }

    gpuErrchk(cudaDeviceSynchronize());

    return 0;
}

extern "C" void helloworld() {
    printf("Hello World\n");
}