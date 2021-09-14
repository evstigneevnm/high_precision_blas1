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
#ifndef __CUMP_BLAS_KERNELS_H__
#define __CUMP_BLAS_KERNELS_H__


#include <cstdint>
#include <cump.h>
#include <vector>
#include <iostream>

template<int BLOCK_SIZE = 1024, int threads_r = 64>
class cump_blas_kernels
{
public:
    cump_blas_kernels(size_t sz_, unsigned int prec_):
    sz(sz_),
    prec(prec_)
    {

    }
    ~cump_blas_kernels()
    {

    }

    void set_prec(unsigned int prec_)
    {
        prec = prec_;
    }

    void dot(const cumpf_array_t& x_device, const cumpf_array_t& y_device, cumpf_array_t& res_device);
    void sum(const cumpf_array_t& x_device, cumpf_array_t& res_device);


    float get_execution_time_milliseconds()
    {
        return(milliseconds);
    }
    
    std::vector<float> get_repeated_execution_time_milliseconds()
    {
        return wall_time;
    }

    void use_benchmark(int repeats_)
    {
        repeats = repeats_;
        wall_time.reserve(repeats);
    }

private:
    size_t sz;
    unsigned int prec;
    std::vector<float> wall_time;
    float milliseconds = 0;
    int repeats = 1;
    void get_blocks_threads_shmem(int n, int maxBlocks, int &blocks, int &threads, int &sdataSize);
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

};



#endif