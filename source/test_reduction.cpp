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

//some test for all implemented vector operations;
#include <cmath>
#include <limits>
#include <iostream>
#include <cstdio>
#include <utils/cuda_support.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_reduction.h>



int main(int argc, char const *argv[])
{
    
    using T = TYPE;
    using gpu_vector_operations_t = gpu_vector_operations<T>;
    using T_vec = gpu_vector_operations_t::vector_type;
    using gpu_reduction_t = gpu_reduction<T, T_vec>;

    if(argc != 3)
    {
        std::cout << argv[0] << " G N; where: " << std::endl;
        std::cout << "  'G' is the GPU PCI-bus number or -1 for selection;" << std::endl;
        std::cout << "  'N' is the vector size. " << std::endl;
        return 0;
    }

    int gpu_pci_id = std::atoi(argv[1]);
    int vec_size = std::atoi(argv[2]);
    init_cuda(gpu_pci_id);

    cublas_wrap *CUBLAS = new cublas_wrap(true);
    gpu_vector_operations_t g_vecs(vec_size, CUBLAS);
    gpu_reduction_t reduction(vec_size);
    T_vec u1_d;
    g_vecs.init_vector(u1_d); g_vecs.start_use_vector(u1_d);
    
    
    for(int j = 0; j<vec_size;j++)
    {
        g_vecs.set_value_at_point(T(j), j, u1_d);
    }
    printf("========================= start =========================\n");
    T res = reduction.sum_debug(u1_d);
    cudaDeviceSynchronize();
    printf("========================= stop  =========================\n");
    std::cout << res << std::endl;

    g_vecs.stop_use_vector(u1_d); g_vecs.free_vector(u1_d);
    return 0;
}