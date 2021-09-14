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
#ifndef __GPU_FILE_OPERATIONS_FUNCTIONS_H__
#define __GPU_FILE_OPERATIONS_FUNCTIONS_H__

#include <utils/cuda_support.h>
#include <common/file_operations.h>

namespace gpu_file_operations_functions
{

    template <class T>
    void write_vector(const std::string &f_name, size_t N, T* vec_gpu, unsigned int prec=16)
    {
        T* vec_cpu = host_allocate<T>(N);
        device_2_host_cpy(vec_cpu, vec_gpu, N);
        file_operations::write_vector<T>(f_name, N, vec_cpu, prec);
        host_deallocate<T>(vec_cpu);
    }

    template <class T>
    void read_vector(const std::string &f_name, size_t N, T*& vec_gpu, unsigned int prec=16)
    {
        T* vec_cpu = host_allocate<T>(N);
        file_operations::read_vector<T>(f_name, N, vec_cpu, prec);
        host_2_device_cpy(vec_gpu, vec_cpu, N);
        host_deallocate<T>(vec_cpu);
    }
    template <class T>
    void write_matrix(const std::string &f_name, size_t Row, size_t Col, T* matrix_gpu, unsigned int prec=16)
    {  
        T* vec_cpu = host_allocate<T>(Row*Col);
        device_2_host_cpy(vec_cpu, matrix_gpu, Row*Col);
        file_operations::write_matrix<T>(f_name, Row, Col, vec_cpu, prec);
        host_deallocate<T>(vec_cpu);
    }

    template <class T>
    void read_matrix(const std::string &f_name, size_t Row, size_t Col, T*& matrix_gpu)
    {
        T* vec_cpu = host_allocate<T>(Row*Col);
        file_operations::read_matrix<T>(f_name, Row, Col, vec_cpu);
        host_2_device_cpy(matrix_gpu, vec_cpu, Row*Col);   
        host_deallocate<T>(vec_cpu);
 
    }


}


#endif
