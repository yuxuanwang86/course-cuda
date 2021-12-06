#include <cstdio>
#include "cuda.h"
#include <iostream>

#define N 1024
#define BSXY 32
#define K 2
float A[N][N];
float B[N][N];
float AA[N][N];
__device__ float dA[N][N];
__device__ float dB[N][N];


__global__ void trans_ker(int n) {
    int row = blockIdx.y * BSXY + threadIdx.y;
    int col = blockIdx.x * BSXY + threadIdx.x;
    int row_t = blockIdx.x * BSXY + threadIdx.y;
    int col_t = blockIdx.y * BSXY + threadIdx.x;
    if (row < n && col < n && row_t < n && col_t < n)
        dB[row_t][col_t] = dA[row][col];
}

__global__ void trans_kerr(int n) {
    int row = blockIdx.y * BSXY + threadIdx.y;
    int col = blockIdx.x * BSXY + threadIdx.x;
    int row_t = blockIdx.x * BSXY + threadIdx.y;
    int col_t = blockIdx.y * BSXY + threadIdx.x;

    __shared__ float sh[BSXY][BSXY + 1];

    if (row < n && col < n)
        sh[threadIdx.y][threadIdx.x] = dA[row][col];
    __syncthreads();

    if (row_t < n && col_t < n) 
        dB[row_t][col_t] = sh[threadIdx.x][threadIdx.y];

}

__global__ void trans_kerrr(int n) {
    int row = blockIdx.y * BSXY * K + threadIdx.y;
    int col = blockIdx.x * BSXY * K + threadIdx.x;
    int row_t = blockIdx.x * BSXY * K + threadIdx.y;
    int col_t = blockIdx.y * BSXY * K + threadIdx.x;

    __shared__ float sh[BSXY * K][BSXY * K + 1];

    for (int i = 0; i < BSXY * K; i+=K) {
        for (int j = 0; j < BSXY * K; j+=K)
            if (row + i < n && col + j < n && threadIdx.y + i < BSXY * K && threadIdx.x + j < BSXY * K)
                sh[threadIdx.y + i][threadIdx.x + j] = dA[row + i][col + j];
    }
    __syncthreads();
    for (int i = 0; i < BSXY * K; i+=K) {
        for (int j = 0; j < BSXY * K; j+=K)
            if (row_t + i < n && col_t + j < n && threadIdx.x + i < BSXY * K && threadIdx.y + j < BSXY * K)
                dB[row_t + i][col_t + j] = sh[threadIdx.x + i][threadIdx.y + j];
    }

}

__global__ void trans_kerrrr(int n) {
    int row = blockIdx.y * BSXY + threadIdx.y;
    int col = blockIdx.x * BSXY + threadIdx.x;
    int row_t = blockIdx.x * BSXY + threadIdx.y;
    int col_t = blockIdx.y * BSXY + threadIdx.x;
    if (col <= row) return;
    if (col_t >= row_t) return;
    __shared__ float shu[BSXY][BSXY + 1];
    __shared__ float shd[BSXY][BSXY + 1];
    if (row < n && col < n)
        shu[threadIdx.y][threadIdx.x] = dA[row][col];
    if (row_t < n && col_t < n)
        shd[threadIdx.y][threadIdx.x] = dA[row_t][col_t];
    __syncthreads();
    dA[row][col] = shu[threadIdx.y][threadIdx.x];
    dA[row_t][col_t] = shd[threadIdx.x][threadIdx.y];


}

int main() {
    // Initialisation
    for (int i = 0; i < N; i++) { 
        for (int j = 0; j < N; j++) { 
            A[i][j] = i + j;
            B[i][j] = 0;
            AA[i][j] = i + j;
        }
    }

    // Copier les tableaux A et B vers le GPU
    // A FAIRE ...
    cudaMemcpyToSymbol(dA, A, N * N * sizeof(float), 0,
            cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dB, B, N * N * sizeof(float), 0,
            cudaMemcpyHostToDevice);

    // Appeler chaque kernel GPU de maniere appropriee pour multiplier les matrices A et B
    // A FAIRE ...
    dim3 dimGrid;
    dimGrid.x = (N - 1) / 32 + 1;
    dimGrid.y = (N - 1) / 32 + 1;
    dimGrid.z = 1;
    dim3 dimBlock;
    dimBlock.x = 32;
    dimBlock.y = 32;
    dimBlock.z = 1;
    trans_kerrrr<<<dimGrid, dimBlock>>>(N);
    cudaMemcpyFromSymbol(A, dA, N * N * sizeof(float), 0, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if (AA[i][j] != A[j][i]) {
                printf("index %d %d error\n", i, j);
                return 1;
            }
        }
    }
    printf("Hello, world!\n");
    return 0;
}
