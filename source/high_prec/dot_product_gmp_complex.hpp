// https://doc.lagout.org/science/0_Computer%20Science/3_Theory/Handbook%20of%20Floating%20Point%20Arithmetic.pdf

#ifndef __DOT_PRODUCT_GMP_COMPLEX_HPP__
#define __DOT_PRODUCT_GMP_COMPLEX_HPP__

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <gmpxx.h>
#include <thrust/complex.h>


/**
 * @brief      This class describes a variable precision dot product of complex floating point vectors using GMP.
 *
 * @tparam     T      { basic floating point type i.e. float or double }
 * @tparam     T_vec  { vector type with the basic complex type induced by basic type T }
 */
template<class T, class T_vec>
class dot_product_gmp_complex
{

using TC = thrust::complex<T>;
// using TC_HP = thrust::complex<mpf_class>; // dosn't work due to forced allign! bug in thrust!
using TC_STL = std::complex<mpf_class>;
using TR = mpf_class;
private:
    T_vec X;
    T_vec Y;
    bool array_set = false;
    unsigned int exact_prec_bits;
    size_t N;

    TC_STL dot_m_std;
    mpf_class dot_real;
    mpf_class dot_imag;
    //TC_HP dot_m;
    T_vec dot_s;



public:
    dot_product_gmp_complex(unsigned int exact_prec_bits_ = 512):
    exact_prec_bits(exact_prec_bits_)
    {
        mpf_set_default_prec(exact_prec_bits);
    }
    
    ~dot_product_gmp_complex()
    {
    }

    void set_arrays(size_t N_, T_vec& input_array_1_, T_vec& input_array_2_)
    {

        X = input_array_1_;
        Y = input_array_2_;

        N = N_;
        array_set = true;
    }
    
    TC dot_exact()
    {

        dot_m_std = mpf_class(0, exact_prec_bits);
        dot_real = mpf_class(0, exact_prec_bits);
        dot_imag = mpf_class(0, exact_prec_bits);

        for(size_t j=0;j<N;j++)
        {
            
            mpf_class x_real(X[j].real(), exact_prec_bits);
            mpf_class x_imag(X[j].imag(), exact_prec_bits);
            mpf_class y_real(Y[j].real(), exact_prec_bits);
            mpf_class y_imag(Y[j].imag(), exact_prec_bits);

            dot_real = dot_real + x_real*y_real + x_imag*y_imag;
            dot_imag = dot_imag + x_real*y_imag - x_imag*y_real;
        }
        dot_m_std = TC_STL(dot_real, dot_imag);
        TC dot_exact_T = TC(dot_real.get_d(),  dot_imag.get_d() );
        
        return TC(dot_exact_T);
    }
    
    void print_res()
    {
        //long double prec.
        std::cout << std::scientific << std::setprecision(128) << dot_m_std << std::endl;
    }
    TC get_error(const TC approx_res_)
    {
        mpf_class ap_res_real = mpf_class(approx_res_.real(), exact_prec_bits);
        mpf_class ap_res_imag = mpf_class(approx_res_.imag(), exact_prec_bits);

        TC_STL dot_v_m( ap_res_real, ap_res_imag );
        TC_STL errC_m = dot_v_m - dot_m_std;

        return TC( std::abs(errC_m.real().get_d()), std::abs(errC_m.imag().get_d()) );
    }

    TC get_error_relative(const TC& approx_res_)
    {
        mpf_class ap_res_real = mpf_class(approx_res_.real(), exact_prec_bits);
        mpf_class ap_res_imag = mpf_class(approx_res_.imag(), exact_prec_bits);

        TC_STL dot_v_m(ap_res_real, ap_res_imag);
        TC_STL errC_m = dot_v_m - dot_m_std;
        mpf_class err_re = errC_m.real();
        if( abs(dot_m_std.real())>0 )
        {
            err_re = errC_m.real()/dot_m_std.real();
        }
        else
        {
            err_re = 0;
        }

        mpf_class err_im = errC_m.imag();
        if(abs(dot_m_std.imag())>0 )
        {
            err_im = errC_m.imag()/dot_m_std.imag();
        }
        else
        {
            err_im = 0;
        }
        
        TC errThrust = TC( std::abs(err_re.get_d()), std::abs(err_im.get_d()) );
        return errThrust;
    }    
 

};

#endif