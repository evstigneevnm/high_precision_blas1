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

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __CONSTANT_DATA_H__
#define __CONSTANT_DATA_H__

//TODO hmm it's hack to overcome problem with my nvcc toolkit 3.2 where __CUDACC__ macro is defined but not __NVCC__
//need to be deleted (figure out from which version there is __NVCC__ macro)
//another reason to think about manual customization
#ifdef __CUDACC__
#ifndef __NVCC__
#define __NVCC__
#endif
#endif

#include <cstdlib>   
#include <cstring> 
#ifdef __NVCC__
#include <utils/cuda_safe_call.h>
#endif

//NOTE like simple CUDA __constant__ variables this 'constant buffer' shares the same visbility principle:
//it's only visible inside current compiling module

//ISSUE didnot figure out anything clever then just make two copies in defines - one for cuda case and one for pure c++
//ISSUE like in tensor_fields it whould be better to create special macro to manage cuda/noncuda behaviour then 
//simply looking for __NVCC__ define


#define __CONSTANT_BUFFER__CTASTR2(pre,post) pre ## post
#define __CONSTANT_BUFFER__CTASTR(pre,post) __CONSTANT_BUFFER__CTASTR2(pre,post)


#ifdef __NVCC__

#ifdef __CUDA_ARCH__
#define DEFINE_CONSTANT_BUFFER_ACCESS_FUNC(buf_type, buf_name)                          \
    __device__ __host__ buf_type    &buf_name()                                         \
    {                                                                                   \
        return  reinterpret_cast<buf_type&>(__CONSTANT_BUFFER__CTASTR(__,buf_name));    \
    }
#else
#define DEFINE_CONSTANT_BUFFER_ACCESS_FUNC(buf_type, buf_name)                          \
    __device__ __host__ static buf_type    &buf_name()                                  \
    {                                                                                   \
        return  reinterpret_cast<buf_type&>(__CONSTANT_BUFFER__CTASTR(__h_,buf_name));  \
    }
#endif

#define DEFINE_CONSTANT_BUFFER(buf_type, buf_name)                                                  \
    __constant__ struct __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name)                                 \
    {                                                                                               \
        long long buf[sizeof(buf_type)/sizeof(long long)+1];                                        \
    } __CONSTANT_BUFFER__CTASTR(__,buf_name);                                                       \
    static __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name) __CONSTANT_BUFFER__CTASTR(__h_,buf_name);    \
    DEFINE_CONSTANT_BUFFER_ACCESS_FUNC(buf_type, buf_name)


#define COPY_TO_CONSTANT_BUFFER(buf_name, data)                                                                                         \
    do {                                                                                                                                \
        __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name)     _data;                                                                          \
        memcpy( &_data, &data, sizeof(data) );                                                                                          \
        CUDA_SAFE_CALL( cudaMemcpyToSymbol(__CONSTANT_BUFFER__CTASTR(__,buf_name), &_data, sizeof(_data), 0, cudaMemcpyHostToDevice) ); \
        memcpy( &(__CONSTANT_BUFFER__CTASTR(__h_,buf_name)), &data, sizeof(data) );                                                     \
    } while (0)

#else

#define DEFINE_CONSTANT_BUFFER_ACCESS_FUNC(buf_type, buf_name)                          \
    static buf_type    &buf_name()                                                      \
    {                                                                                   \
        return  reinterpret_cast<buf_type&>(__CONSTANT_BUFFER__CTASTR(__h_,buf_name));  \
    }

#define DEFINE_CONSTANT_BUFFER(buf_type, buf_name)                                                  \
    struct __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name)                                              \
    {                                                                                               \
        long long buf[sizeof(buf_type)/sizeof(long long)+1];                                        \
    };                                                                                              \
    static __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name) __CONSTANT_BUFFER__CTASTR(__h_,buf_name);    \
    DEFINE_CONSTANT_BUFFER_ACCESS_FUNC(buf_type, buf_name)


#define COPY_TO_CONSTANT_BUFFER(buf_name, data)                                                     \
    do {                                                                                            \
        memcpy( &(__CONSTANT_BUFFER__CTASTR(__h_,buf_name)), &data, sizeof(data) );                 \
    } while (0)

#endif

#endif
