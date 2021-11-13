#include <iostream>
#include <algorithm>
#include <chrono>
#include <cuda.h>

using namespace std;


// Calcul de saxpy en utilisant 1 thread par bloc, 1 operation par thread
__global__
void saxpyBlocs(const int N, float a, const float* x, float* y)
{
  int idx;
  idx = blockIdx.x;
  if (idx < N) y[idx] = a * x[idx] + y[idx];
}


// Calcul de saxpy en utilisant blockSize threads par bloc, 1 operation par thread
__global__
void saxpyBlocsThreads(const int N, float a, const float* x, float* y)
{
  int idx;
  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = a * x[idx] + y[idx];
}


// Calcul de saxpy en utilisant blockSize threads par bloc et effectuant k operation par thread dans un bloc
__global__
void saxpyBlocsThreadsKops(const int N, float a, const float* x, float* y, const int k)
{
  int idx;
  for (idx = blockIdx.x * blockDim.x * k + threadIdx.x; idx < blockIdx.x * blockDim.x * k + threadIdx.x + k; ++idx) {

    if (idx >= N) break;
    y[idx] = x[idx] * a + y[idx];
  }
}

// Fonction CPU de reference pour l'operation saxpy
void saxpy(const int N, float a, float* x, float* y)
{
  for (int i = 0; i < N; i++) { y[i] = a * x[i] + y[i]; }
}

// Verifier si le resultat dans res[N] correspond a saxpy(N, a, x, y)
void verifySaxpy(float a, float* x, float* y, float* res, int N)
{
  int i;
  for (i = 0; i < N; i++) {
    float temp = a * x[i] + y[i];
    if (std::abs(res[i] - temp) / std::max(1e-6f, temp) > 1e-6) {
      cout << res[i] << " " << temp << endl;
      break;
    }
  }
  if (i == N) {
    cout << "saxpy on GPU is correct." << endl;
  }
  else {
    cout << "saxpy on GPU is incorrect on element " << i << "." << endl;
  }
}


int main(int argc, char** argv)
{
  int blockSize;
  int k;
  float* x, * y, * res, * dx, * dy;
  float a = 2.0f;

  int N;

  if (argc < 2) {
    printf("Utilisation: ./saxpy N\n");
    return 0;
  }
  N = atoi(argv[1]);

  // Allouer et initialiser les vecteurs x, y et res sur le CPU
  x = (float*)malloc(N * sizeof(float));
  y = (float*)malloc(N * sizeof(float));
  res = (float*)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    x[i] = i;
    y[i] = 1.0f;
  }

  // Allouer les vecteurs dx[N] et dy[N] sur le GPU, puis copier x et y dans dx et dy.
  cudaMalloc(&dx, sizeof(float) * N);
  cudaMalloc(&dy, sizeof(float) * N);
  cudaMemcpy(dx, x, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dy, y, sizeof(float) * N, cudaMemcpyHostToDevice);
  // Lancer le kernel saxpyBlocs avec un nombre de bloc approprie
  saxpyBlocs << <N, 1 >> > (N, a, dx, dy);
  // Copier dy[N] dans res[N] pour la verification sur CPU
  cudaMemcpy(res, dy, sizeof(float) * N, cudaMemcpyDeviceToHost);