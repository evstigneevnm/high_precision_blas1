/*
 * MIT License
 *
 * Copyright (c) 2020 Evstigneev Nikolay Mikhaylovitch <evstigneevnm@ya.ru>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef __GPU_REDUCTION_IMPL_OGITA_SHMEM_CUH__
#define __GPU_REDUCTION_IMPL_OGITA_SHMEM_CUH__

#include <thrust/complex.h>

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

//Dynamic shared memory specialization
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
template<>
struct __GPU_REDUCTION_OGITA_H__SharedMemory< thrust::complex<float> >
{
    __device__ inline operator       thrust::complex<float> *()
    {
        extern __shared__ thrust::complex<float>(__smem_C[]);
        return (thrust::complex<float> *) __smem_C;
    }

    __device__ inline operator const thrust::complex<float> *() const
    {
        extern __shared__ thrust::complex<float>(__smem_C[]);
        return (thrust::complex<float> *) __smem_C;
    }
};
template<>
struct __GPU_REDUCTION_OGITA_H__SharedMemory< thrust::complex<double> >
{
    __device__ inline operator       thrust::complex<double> *()
    {
        extern __shared__ thrust::complex<double>(__smem_Z[]);
        return (thrust::complex<double> *) __smem_Z;
    }

    __device__ inline operator const thrust::complex<double> *() const
    {
        extern __shared__ thrust::complex<double>(__smem_Z[]);
        return (thrust::complex<double> *) __smem_Z;
    }
};



}
#endif