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
#ifndef __SCFD_COMPLEX_CALLS_WRAP_H__
#define __SCFD_COMPLEX_CALLS_WRAP_H__

#include<thrust/complex.h>
#include<complex>

namespace numerical_algos
{
namespace lin_solvers 
{
namespace detail
{
    using sCf = typename std::complex<float>;
    using sCd = typename std::complex<double>;
    using tCf = typename thrust::complex<float>;
    using tCd = typename thrust::complex<double>;

    template<typename T>
    struct complex_base
    {
        using type = T;
    };
    template<>
    struct complex_base< sCf >
    {
        using type = float;
    };
    template<>
    struct complex_base< sCd >
    {
        using type = double;
    };     
    template<>
    struct complex_base< tCf >
    {
        using type = float;
    };
    template<>
    struct complex_base< tCd >
    {
        using type = double;
    };    



    template<class T>
    bool isnan(T val_)
    {
        return(std::isnan(val_));
    }
    template<>
    bool isnan(sCd val_)
    {
        return (std::isnan(val_.real()))||(std::isnan(val_.imag()));
    }
    template<>
    bool isnan(sCf val_)
    {
        return (std::isnan(val_.real()))||(std::isnan(val_.imag()));
    }
    bool isnan(tCd val_)
    {
        return (std::isnan(val_.real()))||(std::isnan(val_.imag()));
    }
    template<>
    bool isnan(tCf val_)
    {
        return (std::isnan(val_.real()))||(std::isnan(val_.imag()));
    }
    template<class T>
    bool isinf(T val_)
    {
        return(std::isinf(val_));
    }
    template<>
    bool isinf(sCd val_)
    {
        return (std::isinf(val_.real()))||(std::isinf(val_.imag()));
    }
    template<>
    bool isinf(sCf val_)
    {
        return (std::isinf(val_.real()))||(std::isinf(val_.imag()));
    }
    bool isinf(tCd val_)
    {
        return (std::isinf(val_.real()))||(std::isinf(val_.imag()));
    }
    template<>
    bool isinf(tCf val_)
    {
        return (std::isinf(val_.real()))||(std::isinf(val_.imag()));
    }
    template<class T>
    typename complex_base<T>::type abs(T val_)
    {
        return(std::abs(val_));
    }    
    template<>
    typename complex_base<tCf>::type abs(tCf val_)
    {
        return( std::abs< complex_base<tCf>::type >( sCf( val_.real(),val_.imag() ) ) );
    }
    template<>
    typename complex_base<tCd>::type abs(tCd val_)
    {
        return( std::abs< complex_base<tCd>::type >( sCd( val_.real(),val_.imag() ) ) );
    }

    template<class T>
    typename complex_base<T>::type norm_number(T val_)
    {
        return(std::abs(val_));
    }
    template<>
    float norm_number(float val_)
    {
        return(std::abs(val_));
    }
    template<>
    double norm_number(double val_)
    {
        return(std::abs(val_));
    }
    template<>
    typename complex_base<sCf>::type norm_number(sCf val_)
    {
        return(std::abs< complex_base<sCf>::type >(val_));
    }
    template<>
    typename complex_base<sCd>::type norm_number(sCd val_)
    {
        return(std::abs< complex_base<sCd>::type >(val_));
    }
    typename complex_base<tCf>::type norm_number(tCf val_)
    {
        return( std::abs< complex_base<sCf>::type >( sCf( val_.real(),val_.imag() ) ) );
    }
    template<>
    typename complex_base<tCd>::type norm_number(tCd val_)
    {
        return( std::abs< complex_base<sCd>::type >( sCd( val_.real(),val_.imag() ) ) );
    }

    template<class T>
    T sqrt(T val_)
    {
        return(std::sqrt(val_));
    }
    template<>
    tCf sqrt(tCf val_)
    {
        sCf val_cst( val_.real(), val_.imag() );
        return(std::sqrt(val_cst));
    }
    template<>
    tCd sqrt(tCd val_)
    {
        sCd val_cst( val_.real(), val_.imag() );
        return(std::sqrt(val_cst));
    }

}
}
}
#endif