#include <cstdio>
#include <iostream>
#include "cuda.h"

using namespace std;

#define N 1024

// Definir un tableau de float Ad[N] de taille statique sur le GPU
__device__ float dA[N];

int main() {
  float A[N], B[N];
  int i;
  // Initialisation
  for (i = 0; i < N; i++) { A[i] = (float)i; }

  // cudaMemcpy de A[N] vers Ad[N]
  cudaMemcpyToSymbol(dA, A, sizeof(float) * N, 0, cudaMemcpyHostToDevice);

  // cudaMemcpy de Ad[N] vers B[N]
  cudaMemcpyFromSymbol(B, dA, sizeof(float) * N, 0, cudaMemcpyDeviceToHost); 
  // Attendre que les kernels GPUs terminent
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Verifier le resultat
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "La copie est incorrecte!\n"; }
  else { cout << "La copie est correcte!\n"; }

  return 0;
}
