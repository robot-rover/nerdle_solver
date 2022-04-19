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

__global__ void generate_clue_kernel(const uint8_t *secrets, const uint8_t *guess, uint8_t *clues, uint32_t pitch, uint32_t num_batches)
{
    __shared__ uint8_t myGuess[NUM_SLOTS];
    uint8_t myClue[NUM_SLOTS];
    uint8_t counts[SYMBOL_ORD];

    uint32_t batch = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadIdx.x == 0) {
        for (int i = 0; i < NUM_SLOTS; i++) {
            myGuess[i] = guess[i];
        }
    }
    __syncthreads();

    if (batch < num_batches)
    {
        for (int i = 0; i < SYMBOL_ORD; i++)
        {
            counts[i] = 0;
        }
        for (int i = 0; i < NUM_SLOTS; i++)
        {
            // uint8_pitch = pitch / sizeof(uint8_t) => pitch
            uint8_t symbol = secrets[batch * pitch + i];
            uint8_t clue_symbol = symbol == myGuess[i] ? 0 : 1;
            counts[symbol] += clue_symbol;
            myClue[i] = clue_symbol;
        }
        for (int i = 0; i < NUM_SLOTS; i++)
        {
            uint8_t guess_symbol = myGuess[i];
            uint8_t make_yellow = myClue[i] & (counts[guess_symbol] > 0 ? 1 : 0);
            counts[guess_symbol] -= make_yellow;
            myClue[i] += make_yellow;
        }
        for (int i = 0; i < NUM_SLOTS; i++)
        {
            clues[batch * pitch + i] = myClue[i];
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

extern "C" uint8_t *generate_clueg(uint8_t *secrets, uint8_t *guess, uint64_t num_batches)
{
    // Size of host secrets (and retval clues)
    size_t size = num_batches * sizeof(uint8_t) * NUM_SLOTS;
    uint8_t *clues = (uint8_t *)malloc(size);

    // Create secrets on device (2D)
    uint8_t *d_secrets;
    size_t d_pitch;
    cudaMallocPitch((void **)&d_secrets, &d_pitch, sizeof(uint8_t) * NUM_SLOTS, num_batches);

    // Create clues on device (2D)
    uint8_t *d_clues;
    size_t d_pitch_assert;
    cudaMallocPitch((void **)&d_clues, &d_pitch_assert, sizeof(uint8_t) * NUM_SLOTS, num_batches);

    // Create guess on device
    uint8_t *d_guess;
    cudaMalloc((void**)&d_guess, sizeof(uint8_t) * NUM_SLOTS);

    // The arrays must be pitched identically
    assert(d_pitch == d_pitch_assert);

    // Copy secrets (2D) to the device
    size_t h_width = sizeof(uint8_t) * NUM_SLOTS;
    cudaMemcpy2D(d_secrets, d_pitch, secrets, h_width, h_width, num_batches, cudaMemcpyHostToDevice);

    // Copy guess to the device
    cudaMemcpy(d_guess, guess, sizeof(uint8_t) * NUM_SLOTS, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_batches + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA: kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    generate_clue_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_secrets, d_guess, d_clues, d_pitch, num_batches);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    cudaMemcpy2D(clues, h_width, d_clues, d_pitch, sizeof(uint8_t) * NUM_SLOTS, num_batches, cudaMemcpyDeviceToHost);

    // Free device global memory
    cudaFree(d_secrets);
    cudaFree(d_clues);
    cudaFree(d_guess);

    return clues;
}
