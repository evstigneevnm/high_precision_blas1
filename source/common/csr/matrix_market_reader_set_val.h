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