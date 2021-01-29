#ifndef __GPU_REDUCTION_IMPL_OGITA_CUH__
#define __GPU_REDUCTION_IMPL_OGITA_CUH__

#include <common/testing/gpu_reduction_ogita.h>

namespace gpu_reduction_ogita_gpu_kernels
{

template<class T>
struct __GPU_REDUCTION_OGITA_H__SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};
template<>
struct __GPU_REDUCTION_OGITA_H__SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};
template<>
struct __GPU_REDUCTION_OGITA_H__SharedMemory<float>
{
    __device__ inline operator       float *()
    {
        extern __shared__ float __smem_f[];
        return (float *)__smem_f;
    }

    __device__ inline operator const float *() const
    {
        extern __shared__ float __smem_f[];
        return (float *)__smem_f;
    }
};


template<class T>
__device__ inline T __GPU_REDUCTION_OGITA_H__two_prod_device(T &t, T a, T b)
{
    T p = a*b;
    t = fma(a, b, -p);
    return p;
}

template<class T>
__device__ inline T __GPU_REDUCTION_OGITA_H__two_sum_device(T &t, T a, T b)
{
    T s = a+b;
    T bs = s-a;
    T as = s-bs;
    t = (b-bs) + (a-as);
    return s;
}


template <class T, class T_vec, unsigned int blockSize, bool nIsPow2>
__global__ void reduce_sum_ogita_kernel(const T_vec g_idata, T_vec g_odata, int n)
{
//    T *sdata = __GPU_REDUCTION_OGITA_H__SharedMemory<T>();
//    T *cdata = __GPU_REDUCTION_OGITA_H__SharedMemory<T>();

    __shared__ T sdata[1024]; //testing
    __shared__ T cdata[1024];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T main_sum = T(0.0);
    T error_sum = T(0.0);
    T error_local;
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        //main_sum += g_idata[i];
        main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, g_idata[i]);
        error_sum += error_local;

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
        {
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_local, main_sum, g_idata[i+blockSize]);
            error_sum += error_local;            
            //main_sum += g_idata[i+blockSize];
        }
        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = main_sum;
    cdata[tid] = error_sum;
    __syncthreads();
    // printf("%le,%le\n", sdata[tid], cdata[tid]);

    // do reduction in shared mem
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            T error_l1;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l1, main_sum, sdata[tid + 512]);
            sdata[tid] = main_sum;

            T error_l2;
            error_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 512]);
            cdata[tid] = error_sum + error_l1 + error_l2;

        }

        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            // main_sum = main_sum + sdata[tid + 256];
            // sdata[tid] = main_sum;
            T error_l1;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l1, main_sum, sdata[tid + 256]);
            sdata[tid] = main_sum;

            T error_l2;
            error_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 256]);
            cdata[tid] = error_sum + error_l1 + error_l2;
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            // main_sum = main_sum + sdata[tid + 128];
            // sdata[tid] = main_sum;
            T error_l1;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l1, main_sum, sdata[tid + 128]);
            sdata[tid] = main_sum;

            T error_l2;
            error_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 128]);
            cdata[tid] = error_sum + error_l1 + error_l2;
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            // main_sum = main_sum + sdata[tid +  64];
            // sdata[tid] = main_sum;
            T error_l1;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l1, main_sum, sdata[tid + 64]);
            sdata[tid] = main_sum;

            T error_l2;
            error_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cdata[tid + 64]);
            cdata[tid] = error_sum + error_l1 + error_l2;        
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;
        volatile T *cmem = cdata;

        if (blockSize >=  64)
        {
            // main_sum = main_sum + smem[tid + 32];
            // smem[tid] = main_sum;
            T error_l1;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l1, main_sum, smem[tid + 32]);
            smem[tid] = main_sum;

            T error_l2;
            error_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 32]);
            cmem[tid] = error_sum + error_l1 + error_l2;          
        }

        if (blockSize >=  32)
        {
            // main_sum = main_sum + smem[tid + 16];
            // smem[tid] = main_sum;
            T error_l1;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l1, main_sum, smem[tid + 16]);
            smem[tid] = main_sum;

            T error_l2;
            error_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 16]);
            cmem[tid] = error_sum + error_l1 + error_l2;         
        }

        if (blockSize >=  16)
        {
            // main_sum = main_sum + smem[tid +  8];
            // smem[tid] = main_sum;
            T error_l1;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l1, main_sum, smem[tid + 8]);
            smem[tid] = main_sum;

            T error_l2;
            error_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 8]);
            cmem[tid] = error_sum + error_l1 + error_l2;         
        }

        if (blockSize >=   8)
        {
            // main_sum = main_sum + smem[tid +  4];
            // smem[tid] = main_sum;
            T error_l1;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l1, main_sum, smem[tid + 4]);
            smem[tid] = main_sum;

            T error_l2;
            error_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 4]);
            cmem[tid] = error_sum + error_l1 + error_l2;         
        }

        if (blockSize >=   4)
        {
            // main_sum = main_sum + smem[tid +  2];
            // smem[tid] = main_sum;
            T error_l1;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l1, main_sum, smem[tid + 2]);
            smem[tid] = main_sum;

            T error_l2;
            error_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 2]);
            cmem[tid] = error_sum + error_l1 + error_l2;         
        }

        if (blockSize >=   2)
        {
            // main_sum = main_sum + smem[tid +  1];
            // smem[tid] = main_sum;
            T error_l1;
            main_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l1, main_sum, smem[tid + 1]);
            smem[tid] = main_sum;

            T error_l2;
            error_sum = __GPU_REDUCTION_OGITA_H__two_sum_device(error_l2, error_sum, cmem[tid + 1]);
            cmem[tid] = error_sum + error_l1 + error_l2;         
        }
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] =  cdata[0];
       //printf("%le,%le\n", sdata[0], cdata[0] );
    }
}

/*
template <class T, class T_vec, unsigned int blockSize, bool nIsPow2>
__global__ void reduce_dot_kernel(const T_vec g_idata1, const T_vec g_idata2, T_vec g_odata, int n)
{
    T *sdata = __GPU_REDUCTION_OGITA_H__SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T main_sum = T(0.0);
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        main_sum += g_idata1[i]*g_idata2[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
        {

            main_sum += g_idata1[i+blockSize]*g_idata2[i+blockSize];
        }
        i += gridSize;
    }
    // each thread puts its local sum into shared memory
    sdata[tid] = main_sum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] = main_sum = main_sum + sdata[tid + 512];
        }

        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = main_sum = main_sum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = main_sum = main_sum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = main_sum = main_sum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = main_sum = main_sum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = main_sum = main_sum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = main_sum = main_sum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = main_sum = main_sum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = main_sum = main_sum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = main_sum = main_sum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}
*/
}

template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::get_blocks_threads_shmem(int n, int maxBlocks, int &blocks, int &threads, int &smemSize)
{

    threads = (n < BLOCK_SIZE*2) ? nextPow2((n + 1)/ 2) : BLOCK_SIZE;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    blocks = (maxBlocks>blocks) ? blocks : maxBlocks;

}



template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::wrapper_reduce_sum(int blocks, int threads, int smemSize, const T_vec InputV, T_vec OutputV, int N)
{

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    if(isPow2(N))
    {
        switch (threads)
        {
            case 1024:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1024, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 512:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 512, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 256:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 256, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 128:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 128, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 64:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 64, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 32:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 32, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 16:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 16, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  8:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 8, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  4:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 4, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  2:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 2, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  1:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            }       
    }
    else
    {
        switch (threads)
        {
            case 1024:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1024, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 512:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 512, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 256:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 256, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 128:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 128, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 64:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 64, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 32:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 32, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case 16:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 16, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  8:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 8, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  4:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 4, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  2:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 2, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
            case  1:
                gpu_reduction_ogita_gpu_kernels::reduce_sum_ogita_kernel<T, T_vec, 1, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
        }       
    }
        
}

/*
template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
void gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::wrapper_reduce_dot(int blocks, int threads, int smemSize, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, int N)
{

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    if(isPow2(N))
    {
        switch (threads)
        {
            case 1024:
                reduce_dot_kernel<T, T_vec, 1024, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 512:
                reduce_dot_kernel<T, T_vec, 512, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 256:
                reduce_dot_kernel<T, T_vec, 256, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 128:
                reduce_dot_kernel<T, T_vec, 128, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 64:
                reduce_dot_kernel<T, T_vec, 64, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 32:
                reduce_dot_kernel<T, T_vec, 32, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 16:
                reduce_dot_kernel<T, T_vec, 16, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  8:
                reduce_dot_kernel<T, T_vec, 8, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  4:
                reduce_dot_kernel<T, T_vec, 4, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  2:
                reduce_dot_kernel<T, T_vec, 2, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  1:
                reduce_dot_kernel<T, T_vec, 1, true><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            }       
    }
    else
    {
        switch (threads)
        {
            case 1024:
                reduce_dot_kernel<T, T_vec, 1024, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 512:
                reduce_dot_kernel<T, T_vec, 512, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 256:
                reduce_dot_kernel<T, T_vec, 256, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 128:
                reduce_dot_kernel<T, T_vec, 128, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 64:
                reduce_dot_kernel<T, T_vec, 64, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 32:
                reduce_dot_kernel<T, T_vec, 32, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case 16:
                reduce_dot_kernel<T, T_vec, 16, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  8:
                reduce_dot_kernel<T, T_vec, 8, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  4:
                reduce_dot_kernel<T, T_vec, 4, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  2:
                reduce_dot_kernel<T, T_vec, 2, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
            case  1:
                reduce_dot_kernel<T, T_vec, 1, false><<< dimGrid, dimBlock, smemSize >>>(InputV1, InputV2, OutputV, N); break;
        }       
    }
        
}

*/

template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
T gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::reduction_sum(int N, const T_vec InputV, T_vec OutputV, T_vec Output)
{
    T gpu_result=0.0;
    int threads = 0, blocks = 0, smemSize=0;
    int maxBlocks=1024;//DEBUG
    get_blocks_threads_shmem(N, maxBlocks, blocks, threads, smemSize);

    //perform reduction
    //printf("threads=%i, blocks=%i, shmem size=%i\n",threads, blocks, smemSize);
    wrapper_reduce_sum(blocks, threads, smemSize, InputV, OutputV, N);
    bool needReadBack=true;
    int s=blocks;
    while (s > 1)
    {
        get_blocks_threads_shmem(s, maxBlocks, blocks, threads, smemSize);
        //printf("threads=%i, blocks=%i, shmem size=%i\n",threads, blocks, smemSize);
        wrapper_reduce_sum(blocks, threads, smemSize, OutputV, OutputV, s);
        s = (s + (threads*2-1)) / (threads*2);
    }
    if (s > 1)
    {
        //cudaMemcpy(Output, OutputV, s * sizeof(T), cudaMemcpyDeviceToHost);
        device_2_host_cpy<T>(Output, OutputV, s);

        for (int i=0; i < s; i++)
        {
            gpu_result += Output[i];
        }
        needReadBack = false;
    }
    if (needReadBack)
    {
        //cudaMemcpy(&gpu_result, OutputV, sizeof(T), cudaMemcpyDeviceToHost);
        device_2_host_cpy<T>(&gpu_result, OutputV, 1);
    }
    return gpu_result;  
}


/*
template<class T, class T_vec, int BLOCK_SIZE, int threads_r>
T gpu_reduction_ogita<T, T_vec, BLOCK_SIZE, threads_r>::reduction_dot(int N, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, T_vec Output)
{
    T gpu_result=0.0;
    int threads = 0, blocks = 0, smemSize=0;
    int maxBlocks=1024;//DEBUG
    get_blocks_threads_shmem(N, maxBlocks, blocks, threads, smemSize);

    //perform reduction
    //printf("threads=%i, blocks=%i, shmem size=%i\n",threads, blocks, smemSize);
    wrapper_reduce_dot(blocks, threads, smemSize, InputV1, InputV2, OutputV, N);
    bool needReadBack=true;
    int s=blocks;
    while (s > 1)
    {
        get_blocks_threads_shmem(s, maxBlocks, blocks, threads, smemSize);
        //printf("threads=%i, blocks=%i, shmem size=%i\n",threads, blocks, smemSize);
        wrapper_reduce_sum(blocks, threads, smemSize, OutputV, OutputV, s);
        s = (s + (threads*2-1)) / (threads*2);
    }
    if (s > 1)
    {
        //cudaMemcpy(Output, OutputV, s * sizeof(T), cudaMemcpyDeviceToHost);
        device_2_host_cpy<T>(Output, OutputV, s);

        for (int i=0; i < s; i++)
        {
            gpu_result += Output[i];
        }
        needReadBack = false;
    }
    if (needReadBack)
    {
        //cudaMemcpy(&gpu_result, OutputV, sizeof(T), cudaMemcpyDeviceToHost);
        device_2_host_cpy<T>(&gpu_result, OutputV, 1);
    }
    return gpu_result;  
}
*/


#endif