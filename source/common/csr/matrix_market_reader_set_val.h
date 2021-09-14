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
#ifndef __CSR__MATRIX_MARKET_READER_SET_VAL_H__
#define __CSR__MATRIX_MARKET_READER_SET_VAL_H__


#include<complex>
#include<thrust/complex.h>
//TODO: use traits?

namespace csr
{
    
    template<class T>
    struct complex_base_type
    {
        using real = T;
        
    };
    template<>
    struct complex_base_type<std::complex<float> >
    {
        using real = float;


    };
    template<>
    struct complex_base_type<std::complex<double> >
    {
        using real = double;
      
    };
    template<>
    struct complex_base_type<thrust::complex<float> >
    {
        using real = float;
        
    };
    template<>
    struct complex_base_type<thrust::complex<double> >
    {
        using real = double;       
    };

    template<class Tl>
    void set_val(Tl& out_, double in_1_, double in_2_)
    {
        out_ = in_1_;
    }
    template<>
    void set_val(thrust::complex<float>& out_, double in_1_, double in_2_)
    {
        out_ = thrust::complex<float>(in_1_, in_2_);
    }
    template<>
    void set_val(thrust::complex<double>& out_, double in_1_, double in_2_)
    {
        out_ = thrust::complex<double>(in_1_, in_2_);
    }
    template<>
    void set_val(std::complex<float>& out_, double in_1_, double in_2_)
    {
        out_ = std::complex<float>(in_1_, in_2_);
    }
    template<>
    void set_val(std::complex<double>& out_, double in_1_, double in_2_)
    {
        out_ = std::complex<double>(in_1_, in_2_);
    }

}
#endif