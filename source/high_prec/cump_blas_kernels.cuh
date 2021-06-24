/*
 *  Multiple-precision BLAS routines using CUMP as well as corresponding performance benchmarks.
 *
 *  Copyright 2018, 2019 by Konstantin Isupov and Alexander Kuvaev.
 *
 *  This file is part of the MPRES-BLAS library.
 *
 *  MPRES-BLAS is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  MPRES-BLAS is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with MPRES-BLAS.  If not, see <https://www.gnu.org/licenses/>.
 */
// Taken from: https://github.com/kisupov/mpres-blas
// The library itself cannot compile due to some errors.
// 


#ifndef __CUMP_BLAS_KERNELS_CUH__
#define __CUMP_BLAS_KERNELS_CUH__

#include <stdio.h>
#include <iostream>
#include <cump/cump.cuh>
#include <high_prec/cump_blas_kernels.h>


namespace kernels_from_mpblas
{
/********************* Computational kernels *********************/

/*
 * Computes the sum of the elements of vector x
 */
 template<int id = 0>
__global__ void cump_sum_kernel1(int n, cump::mpf_array_t result, cump::mpf_array_t x, cump::mpf_array_t temp){
    using namespace cump;
    // parameters
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int globalIdx = bid * bsize + tid;
    unsigned int i = bid * bsize * 2 + tid;
    unsigned int k = 2 * gridDim.x * bsize;

    while (i < n)
    {
        mpf_add(temp[bid * bsize + tid], temp[bid * bsize + tid], x[i]);
        if (i + bsize < n)
        {
            cump::mpf_add(temp[globalIdx], temp[globalIdx], x[i + bsize]);
        }
        i += k;
        
    }
    __syncthreads();

    i = bsize;
    while(i >= 2)
    {
        unsigned int half = i >> 1;
        if ((bsize >= i) && (tid < half) && (globalIdx + half < n))
        {
            cump::mpf_add(temp[globalIdx], temp[globalIdx], temp[globalIdx + half]);
        }
        i = i >> 1;
        __syncthreads();
    }
    
    if (tid == 0)
    {    
        cump::mpf_set(result[bid], temp[globalIdx]);
    };
    __syncthreads();
}

/*
 * Computes the sum of the elements of vector x (optimized kernel)
 */
__global__ void cump_sum_kernel2(cump::mpf_array_t x, cump::mpf_array_t result){
    unsigned int tid = threadIdx.x;
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s)
        {
            cump::mpf_add(x[tid], x[tid], x[tid + s]);
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
    {
        cump::mpf_set(result[0], x[tid]);
    }
}

/*
 * Computes the element-wise vector-vector product
 */
__global__ void cump_vec_mul_kernel(int n, cump::mpf_array_t result, cump::mpf_array_t x, cump::mpf_array_t y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=n)
    {
        return;
    }
    cump::mpf_mul(result[idx], y[idx], x[idx]);
    //idx += gridDim.x * blockDim.x;
    
}

/*
 * Multiplies a scalar by a vector
 */
__global__ void cump_scal_kernel(int n, cump::mpf_array_t alpha, cump::mpf_array_t x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        cump::mpf_mul(x[idx], alpha[0], x[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

/*
 * Constant times a vector plus a vector
 */
__global__  void cump_axpy_kernel(int n, cump::mpf_array_t a, cump::mpf_array_t X, cump::mpf_array_t Y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        cump::mpf_mul(X[idx], a[0], X[idx]);
        cump::mpf_add(Y[idx], X[idx], Y[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

/*
 * Performs rotation of points in the plane
 */
__global__  void cump_rot_kernel(int n, cump::mpf_array_t x, cump::mpf_array_t y, cump::mpf_array_t c, cump::mpf_array_t s, cump::mpf_array_t buffer1, cump::mpf_array_t buffer2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        //perform c * x
        cump::mpf_mul(buffer1[idx], c[0], x[idx]);
        //perform s * y
        cump::mpf_mul(buffer2[idx], s[0], y[idx]);
        //perform y = c * y - s * x
        cump::mpf_mul(x[idx], x[idx], s[0]);
        cump::mpf_mul(y[idx], y[idx], c[0]);
        cump::mpf_sub(y[idx], y[idx], x[idx]);
        //perform x = c * x + s * y
        cump::mpf_add(x[idx], buffer1[idx], buffer2[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

/*
 * Performs the matrix-vector operation  y := A*x + beta*y,
 * where beta is a scalar, x and y are vectors and A is an m by n matrix
 */
__global__ void cump_gemv_kernel(int m, int n, cump::mpf_array_t alpha, cump::mpf_array_t A, int lda, cump::mpf_array_t x, cump::mpf_array_t beta, cump::mpf_array_t y, cump::mpf_array_t tmp1)  {
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId < m) {
        cump::mpf_mul(y[threadId], beta[0], y[threadId]);
        for (int colId = 0; colId < n; colId++) {
            cump::mpf_mul(tmp1[threadId], x[colId], A[colId * lda + threadId]);
            cump::mpf_add(y[threadId], y[threadId], tmp1[threadId]);
        }
    }
}

/*
* Computes a matrix-matrix product with general matrices.
* C = alpha * A * B + beta * C
* where alpha and beta are scalars, A, B, and C are matrices.
* All the matrices should be stored in column-major order.
 */
__global__ void cump_gemm_kernel(int m, int n, int k, cump::mpf_array_t alpha, cump::mpf_array_t A, int lda, cump::mpf_array_t B, int ldb, cump::mpf_array_t beta, cump::mpf_array_t C, int ldc, cump::mpf_array_t buf1, cump::mpf_array_t buf2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexC = row + col * ldc;
    if(col < n && row < m){
        for(int i = 0; i < k; i++){
            cump::mpf_mul(buf1[indexC], alpha[0], A[lda * i + row]);
            cump::mpf_mul(buf1[indexC], B[col * ldb + i], buf1[indexC]);
            cump::mpf_add(buf2[indexC], buf1[indexC], buf2[indexC]);
        }
        cump::mpf_mul(C[indexC], beta[0], C[indexC]);
        cump::mpf_add(C[indexC], buf2[indexC], C[indexC]);
    }
}

/*
 * Performs the matrix-vector operation  A := x*y^T + A,
 * x and y are vectors and A is an lda by n matrix
 */
__global__ void cump_ger_kernel(int m, int n, cump::mpf_array_t A, int lda, cump::mpf_array_t x, cump::mpf_array_t y, cump::mpf_array_t tmp1) {
    int j = blockIdx.y; // The column index
    while (j < n){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if( i < m ){
            cump::mpf_mul(tmp1[i + j * m], x[i], y[j]);
            cump::mpf_add(A[i + j * lda], A[i + j * lda], tmp1[i + j * m]);
            i += gridDim.x * blockDim.x;
        }
        __syncthreads();
        j += gridDim.y;
    }
}

/*
* Scales two matrices A and B and stores their sum in a matrix C
* C = alpha * A + beta * B
* where alpha and beta are scalars, and A, B, C are m by n matrices.
* All the matrices should be stored in column-major order.
 */
__global__ void cump_ge_add_kernel(int m, int n, cump::mpf_array_t alpha, cump::mpf_array_t A, int lda, cump::mpf_array_t beta, cump::mpf_array_t B, int ldb, cump::mpf_array_t C, int ldc, cump::mpf_array_t buf) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < m) {
        cump::mpf_mul(C[row + col * ldc], beta[0], B[row + col * ldb]);
        cump::mpf_mul(buf[row + col * m], alpha[0], A[row + col * lda]);
        cump::mpf_add(C[row + col * ldc], buf[row + col * m], C[row + col * ldc]);
    }
}

/*
* Scales a matrix A and scales a matrix B and accumulates the result in the matrix B
* B = alpha * A + beta * B
* where alpha and beta are scalars, and A and B are matrices.
* All the matrices should be stored in column-major order.
 */
__global__ void cump_ge_acc_kernel(int m, int n, cump::mpf_array_t alpha, cump::mpf_array_t A, int lda, cump::mpf_array_t beta, cump::mpf_array_t B, int ldb, cump::mpf_array_t buf) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < m) {
        cump::mpf_mul(B[row + col * ldb], beta[0], B[row + col * ldb]);
        cump::mpf_mul(buf[row + col * m], alpha[0], A[row + col * lda]);
        cump::mpf_add(B[row + col * ldb], buf[row + col * m], B[row + col * ldb]);
    }
}


/*
* Scales a general matrix A on the right side or by a diagonal matrix D: A = AD
 */
__global__ void cump_ge_diag_scale_r_kernel(int m, int n, cump::mpf_array_t D, int incd, cump::mpf_array_t A, int lda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexA = row + col * lda;
    int indexD = incd > 0 ? col * incd : (-n + col + 1)*incd;
    if (col < n && row < m) {
        cump::mpf_mul(A[indexA], A[indexA], D[indexD]);
    }
}

/*
* Scales a general matrix A on the left side or by a diagonal matrix D: A = DA
 */
__global__ void cump_ge_diag_scale_l_kernel(int m, int n, cump::mpf_array_t D, int incd, cump::mpf_array_t A, int lda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexA = row + col * lda;
    int indexD = incd > 0 ? row * incd : (-m + row + 1)*incd;
    if (col < n && row < m) {
        cump::mpf_mul(A[indexA], A[indexA], D[indexD]);
    }
}

/*
* Scales a general matrix A on the left side by a diagonal matrix DL and on the right side by a diagonal matrix DR: A = DL * A * DR
 */
__global__ void cump_ge_lrscale_kernel(int m, int n, cump::mpf_array_t DL, int incdl, cump::mpf_array_t DR, int incdr, cump::mpf_array_t A, int lda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexA = row + col * lda;
    int indexDL = incdl > 0 ? row * incdl : (-m + row + 1)*incdl;
    int indexDR = incdr > 0 ? col * incdr : (-n + col + 1)*incdr;
    if (col < n && row < m) {
        cump::mpf_mul(A[indexA], A[indexA], DL[indexDL]);
        cump::mpf_mul(A[indexA], A[indexA], DR[indexDR]);
    }
}

/*
 * Set the elements of an array to zero
 */
__global__ void cump_reset_array(int n, cump::mpf_array_t temp) {
    unsigned int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
    while (numberIdx < n) {
        cump::mpf_sub(temp[numberIdx], temp[numberIdx], temp[numberIdx]); // set to zero
        numberIdx +=  gridDim.x * blockDim.x;
    }
}


}

/********************* cpp functions *********************/

template<int BLOCK_SIZE, int threads_r>
void cump_blas_kernels<BLOCK_SIZE, threads_r>::sum(const cumpf_array_t& x, cumpf_array_t& res){


    // int blocks_mul = sz / threads_r + (sz % threads_r ? 1 : 0);
    // int blocks_red = BLOCK_SIZE;


    
    // //Host data
    // mpf_t *hx = new mpf_t[n];
    // mpf_t hresult;

    // //GPU data
    // cucump::mpf_array_t dx;
    // cucump::mpf_array_t dresult;
    // cucump::mpf_array_t dtemp;
    // cucump::mpf_array_t dblock_result;

    // cumpf_array_init2(dx, n, prec);
    // cumpf_array_init2(dresult, 1, prec);
    // cumpf_array_init2(dtemp, n, prec);
    // cumpf_array_init2(dblock_result, blocks, prec);

    // //Convert from MPFR
    // for(int i = 0; i < n; i ++){
    //     mpf_init2(hx[i], prec);
    //     mpf_set_str(hx[i], convert_to_string_sci(x[i], convert_digits).c_str(), 10);
    // }
    // mpf_init2(hresult, prec);
    // mpf_set_d(hresult, 0);

    // //Copying to the GPU
    // cumpf_array_set_mpf(dx, hx, n);

    // //Launch
    // for(int i = 0; i < repeat; i ++){
    //     cump_reset_array<<<blocks, threads>>>(n, dtemp);
    //     StartCudaTimer();
    //     cump_sum_kernel1<<<blocks, threads>>>(n, dblock_result, dx, dtemp);
    //     cump_sum_kernel2<<<1, blocks>>>(dblock_result, dresult);
    //     EndCudaTimer();
    // }
    // PrintCudaTimer("took");

    // //Copying to the host
    // mpf_array_set_cumpf(&hresult, dresult, 1);
    // gmp_printf ("result: %.70Ff \n", hresult);

    // //Cleanup
    // mpf_clear(hresult);
    // for(int i = 0; i < n; i ++){
    //     mpf_clear(hx[i]);
    // }
    // delete[] hx;
    // cumpf_array_clear(dx);
    // cumpf_array_clear(dresult);
    // cumpf_array_clear(dblock_result);
    // cumpf_array_clear(dtemp);
}


template<int BLOCK_SIZE, int threads_r>
void cump_blas_kernels<BLOCK_SIZE, threads_r>::get_blocks_threads_shmem(int n, int maxBlocks, int &blocks, int &threads, int &sdataSize)
{

    unsigned int BLOCK_SIZE_l = 128;
    threads = (n < BLOCK_SIZE_l*2) ? nextPow2((n + 1)/ 2) : BLOCK_SIZE_l;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    blocks = (maxBlocks>blocks) ? blocks : maxBlocks;

}


template<int BLOCK_SIZE, int threads_r>
void cump_blas_kernels<BLOCK_SIZE, threads_r>::dot(const cumpf_array_t& x, const cumpf_array_t& y, cumpf_array_t& res)
{

    const int maxBlocks = std::pow<int>(2,31) - 1;// sm_30 and greater.
    
    cudaEvent_t start, stop;

    int threads = 0, blocks = 0, sdataSize=0;
    get_blocks_threads_shmem(sz, maxBlocks, blocks, threads, sdataSize);
    cumpf_array_t dtemp;
    cumpf_array_t dvecprod;
    cumpf_array_t dblock_result;
    cumpf_array_init2(dtemp, sz, prec);
    cumpf_array_init2(dvecprod, sz, prec);
    cumpf_array_init2(dblock_result, sz, prec);
    

    int blocks_x=std::floor(sz/( BLOCK_SIZE ))+1;

    //Launch
    kernels_from_mpblas::cump_reset_array<<<blocks_x, BLOCK_SIZE>>>(sz, dtemp);
    

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int r=0;r<repeats;r++)
    {
        get_blocks_threads_shmem(sz, maxBlocks, blocks, threads, sdataSize);
        kernels_from_mpblas::cump_vec_mul_kernel<<<blocks_x, BLOCK_SIZE>>>(sz, dvecprod, x, y);
        cudaEventRecord(start); 
        kernels_from_mpblas::cump_sum_kernel1<<<blocks, threads>>>(sz, dblock_result, dvecprod, dtemp);
        int s=blocks;
        while (s > 1)
        {
            get_blocks_threads_shmem(s, maxBlocks, blocks, threads, sdataSize);
            dim3 dimBlock(threads, 1, 1);
            dim3 dimGrid(blocks, 1, 1);
            kernels_from_mpblas::cump_reset_array<<<dimGrid, dimBlock>>>(s, dtemp);
            kernels_from_mpblas::cump_sum_kernel1<<<dimGrid, dimBlock>>>(s, dblock_result, dblock_result, dtemp);
            s = (s + (threads*2-1)) / (threads*2);
        }

        if(s==1)
        {
            kernels_from_mpblas::cump_sum_kernel2<<<1, 1>>>(dblock_result, res);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        if(repeats > 1)
        {
            wall_time.push_back(milliseconds);
        }
    }
    cumpf_array_clear(dblock_result);
    cumpf_array_clear(dvecprod);
    cumpf_array_clear(dtemp);



}



#endif //MPRES_TEST_CUMP_BLAS_CUH