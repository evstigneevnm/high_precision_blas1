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
#ifndef __GPU_REDUCTION_OGITA_TYPE_H__
#define __GPU_REDUCTION_OGITA_TYPE_H__

#include <thrust/complex.h>

namespace gpu_reduction_ogita_type{

template<typename T_>
struct type_complex_cast
{
    using T = T_;
};
  
template<>
struct type_complex_cast< thrust::complex<float> >
{
    using T = float;
};
template<>
struct type_complex_cast< thrust::complex<double> >
{
    using T = double;
};    


template<typename T>
struct return_real
{
    using T_real = typename type_complex_cast<T>::T;
    T_real get_real(T val)
    {
        return val;
    }    
};


template<>
struct return_real< thrust::complex<float> >
{
    using T_real = float;//typename type_complex_cast< thrust::complex<float> >::T;
    T_real get_real(thrust::complex<float> val)
    {
        return val.real();
    }    
};
template<>
struct return_real< thrust::complex<double> >
{
    using T_real = double;//typename type_complex_cast< thrust::complex<double> >::T;
    T_real get_real(thrust::complex<double> val)
    {
        return val.real();
    }    
};

}


#endif

    