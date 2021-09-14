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
#ifndef __GENERATE_VECTOR_PAIR_HELPER_H__
#define __GENERATE_VECTOR_PAIR_HELPER_H__

#include <cstddef>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <utils/cuda_support.h>

template<class T, class T_vec, int BLOCK_SIZE = 1024>
class generate_vector_pair_helper
{
public:
    generate_vector_pair_helper(size_t sz_):
    sz(sz_)
    {
        calculate_cuda_grid();
    }
    ~generate_vector_pair_helper()
    {   

    }
    void generate_C_estimated_vector(const T max_power, const T_vec mantisa_v, const T_vec exponent_v, T_vec return_v, size_t N_fixed_value = 0);

    void return_abs_vec(const T_vec x_in, T_vec x_out);
    void return_abs_vec_inplace(T_vec x_);
    void return_abs_double_vec_inplace(double* x_);
    void convert_vector_T_to_double(T_vec x_T_, double* x_D_);

    void init_double_vectors(double*& x1, double*& x2)
    {
        x1 = device_allocate<double>(sz);
        x2 = device_allocate<double>(sz);
    }
    void delete_double_vectors(double*& x1, double*& x2)
    {
        if(x1 != nullptr)
            device_deallocate<double>(x1);
        if(x2 != nullptr)
            device_deallocate<double>(x2);
    }

private:
    size_t sz;
    dim3 dimBlock;
    dim3 dimGrid;
    void calculate_cuda_grid();

};

    

#endif