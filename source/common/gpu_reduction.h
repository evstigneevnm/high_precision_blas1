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
#ifndef __GPU_REDUCTION_H__
#define __GPU_REDUCTION_H__

#include <utility>
#include <cstddef>
#include <cmath>
#include <utils/cuda_support.h>


#define __GPU_REDUCTION_H__inf 0x7f800000 


template<class T, class T_vec, int BLOCK_SIZE = 1024, int threads_r = 64>
class gpu_reduction
{
public:
    using min_max_t = std::pair<T,T>;

    gpu_reduction(size_t vec_size_):
    vec_size(vec_size_)
    {
        vec_helper = device_allocate_host<T>(vec_size);
        vec_helper_d = device_allocate<T>(vec_size);

    }
    ~gpu_reduction()
    {
        if(vec_helper != nullptr)
        {
            device_deallocate_host(vec_helper);
        }
        if(vec_helper_d != nullptr)
        {
            device_deallocate(vec_helper_d);
        }
    }

    min_max_t min_max(const T_vec d_in)
    {
        compute_reduction_min_max(d_in, vec_helper_d, vec_size);
        min_max_t result;
        device_2_host_cpy<T>(&result.first, &vec_helper_d[0], 1);
        device_2_host_cpy<T>(&result.second, &vec_helper_d[1], 1);
        
        return result;
    }

    T sum(const T_vec d_in)
    {
        T res = reduction_sum(vec_size, d_in, vec_helper_d, vec_helper, false);
        return res;
    }
    T sum_debug(const T_vec d_in)
    {
        T res = reduction_sum_debug(vec_size, d_in, vec_helper_d, vec_helper);
        return res;
    }    
    T asum(const T_vec d_in)
    {
        T res = reduction_sum(vec_size, d_in, vec_helper_d, vec_helper, true);
        return res;
    }
    T dot(const T_vec d1_in, const T_vec d2_in)
    {
        T res = reduction_dot(vec_size, d1_in, d2_in, vec_helper_d, vec_helper);
        return res;
    }

private:
    const int maxBlocks = std::pow<int>(2,31) - 1;// sm_30 and greater.

    size_t vec_size;
    T_vec vec_helper_d = nullptr;
    T_vec vec_helper = nullptr;


    void compute_reduction_min_max(const T_vec d_in, T_vec vec_helper_d, int num_el);

    T reduction_sum(int num_el, const T_vec InputV, T_vec OutputV, T_vec Output, bool use_abs_);
    
    T reduction_sum_debug(int num_el, const T_vec InputV, T_vec OutputV, T_vec Output);

    T reduction_dot(int N, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, T_vec Output);

    void findBlockSize(int* whichSize, int num_el);
    // for any integer returns the closest larger power_of_two neighbour.
    unsigned int nextPow2(unsigned int x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }
    bool isPow2(unsigned int x)
    {
        return ( (x&(x-1))==0 );
    }
    void get_blocks_threads_shmem(int n, int maxBlocks, int& blocks, int& threads, int& smemSize);
    void wrapper_reduce_sum(int blocks, int threads, int smemSize, const T_vec InputV, T_vec OutputV, int N);
    void wrapper_reduce_sum_debug(int blocks, int threads, int smemSize, const T_vec InputV, T_vec OutputV, int N);
    void wrapper_reduce_asum(int blocks, int threads, int smemSize, const T_vec InputV, T_vec OutputV, int N);    
    void wrapper_reduce_dot(int blocks, int threads, int smemSize, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, int N);

};


#endif

    