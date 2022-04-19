#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cassert>

#define NUM_SLOTS 8
#define SYMBOL_ORD 15
__device__ static constexpr char SYMBOL_TABLE[SYMBOL_ORD + 1] = "0123456789+-*/=";

__global__ void vectorAdd(const double *A, const double *B, double *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void generate_clue_kernel(const uint8_t *secret, uint32_t num_secret, uint32_t secret_pitch, uint8_t *guess, uint32_t num_guess, uint32_t guess_pitch, uint8_t *clues, uint32_t clues_pitch)
{
    __shared__ uint8_t secret_cache[32 * NUM_SLOTS];
    __shared__ uint8_t guess_cache[32 * NUM_SLOTS];
    uint8_t counts[SYMBOL_ORD];
    uint8_t clue[NUM_SLOTS];

    uint32_t secret_addr = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t secret_offset = threadIdx.x * NUM_SLOTS;
    uint32_t guess_addr = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t guess_offset = threadIdx.y * NUM_SLOTS;
    int32_t clues_offset = secret_addr * NUM_SLOTS + guess_addr * clues_pitch;

    if (secret_addr < num_secret && guess_addr < num_guess)
    {
        if (threadIdx.y == 0)
        {
            uint32_t x_cache_offset = threadIdx.x * NUM_SLOTS;
            uint32_t x_main_offset = secret_addr * secret_pitch;
            for (int i = 0; i < NUM_SLOTS; i++)
            {
                secret_cache[x_cache_offset + i] = secret[x_main_offset + i];
            }
        }
        if (threadIdx.x == 0)
        {
            uint32_t y_cache_offset = threadIdx.y * NUM_SLOTS;
            uint32_t y_main_offset = guess_addr * guess_pitch;
            for (int i = 0; i < NUM_SLOTS; i++)
            {
                guess_cache[y_cache_offset + i] = guess[y_main_offset + i];
            }
        }
    }

    __syncthreads();

    if (secret_addr < num_secret && guess_addr < num_guess)
    {
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
        }
        for (int i = 0; i < NUM_SLOTS; i++)
        {
            uint8_t guess_symbol = guess_cache[secret_offset + i];
            uint8_t one_if_yellow = clue[i] & (counts[guess_symbol] > 0 ? 1 : 0);
            counts[guess_symbol] -= one_if_yellow;
            clue[i] += one_if_yellow;
        }
        for (int i = 0; i < NUM_SLOTS; i++)
        {
            clues[clues_offset + i] = clue[i];
        }
    }
}

extern "C" double *myVectorAdd(double *h_A, double *h_B, int numElements)
{
    size_t size = numElements * sizeof(double);
    // printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host output vector C
    double *h_C = (double *)malloc(size);

    // Allocate the device input vectors
    double *d_A = NULL;
    cudaMalloc((void **)&d_A, size);

    double *d_B = NULL;
    cudaMalloc((void **)&d_B, size);
    double *d_C = NULL;
    cudaMalloc((void **)&d_C, size);

    // printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    // printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
}

extern "C" void generate_clueg(uint8_t *guess_eqs, uint32_t num_guess, uint8_t *secret_eqs, uint32_t num_secret, uint8_t *clue_arr)
{
    // Size of host eqs (and retval clues)
    size_t eq_width = sizeof(uint8_t) * NUM_SLOTS;
    size_t size_guess = num_guess * eq_width;
    size_t size_secret = num_secret * eq_width;

    // Create guess on device (2D)
    uint8_t *d_guess;
    size_t guess_pitch;
    cudaMallocPitch((void **)&d_guess, &guess_pitch, eq_width, num_guess);

    // Create secret on device (2D)
    uint8_t *d_secret;
    size_t secret_pitch;
    cudaMallocPitch((void **)&d_secret, &secret_pitch, eq_width, num_secret);

    // Create clues on device (2D)
    uint8_t *d_clues;
    size_t clues_pitch;
    cudaMallocPitch((void **)&d_clues, &clues_pitch, eq_width * num_secret, num_guess);

    // Copy to the device
    cudaMemcpy2D(d_secret, secret_pitch, secret_eqs, eq_width, eq_width, num_secret, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_guess, guess_pitch, guess_eqs, eq_width, eq_width, num_guess, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 32;
    int blocksPerGridx = (num_secret + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridy = (num_guess + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blocks(blocksPerGridx, blocksPerGridy);
    dim3 threads(threadsPerBlock, threadsPerBlock);

    // printf("CUDA: kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    generate_clue_kernel<<<blocks, threads>>>(d_secret, num_secret, secret_pitch, d_guess, num_guess, guess_pitch, d_clues, clues_pitch);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    cudaMemcpy2D(clue_arr, eq_width * num_secret, d_clues, clues_pitch, eq_width * num_secret, num_guess, cudaMemcpyDeviceToHost);

    // Free device global memory
    cudaFree(d_secret);
    cudaFree(d_guess);
    cudaFree(d_clues);

    return;
}
