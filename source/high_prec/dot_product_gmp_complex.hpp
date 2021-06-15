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
        for(size_t j=0;j<N;j++)
        {
            
            TC_STL x_l( std::complex<T>(conj(X[j])) );
            TC_STL y_l( std::complex<T>(Y[j]) );
            dot_m_std = dot_m_std + x_l*y_l;
        }
        
        thrust::complex<double> dot_exact_T = TC( dot_m_std.real().get_d(), dot_m_std.imag().get_d() );
        
        return TC(dot_exact_T);
    }
    
    void print_res()
    {
        //long double prec.
        std::cout << std::scientific << std::setprecision(128) << dot_m_std << std::endl;
    }
    T get_error(const TC& approx_res_)
    {
        TC_STL dot_v_m( std::complex<T>(approx_res_.real(), approx_res_.imag()) );
        TC_STL errC_m = dot_v_m - dot_m_std;
        mpf_class err_m = abs(errC_m);

        double err_d = err_m.get_d();
        return T(err_d);
    }
    T get_error_T(const TC& approx_res_)
    {
        TC_STL dot_c = TC_STL(dot_m_std.real().get_d(), dot_m_std.imag().get_d() );
        TC_STL err_c = TC_STL(approx_res_) - dot_c;
        

        return std::abs(err_c);
    }
    TC get_error_relative(const TC& approx_res_)
    {
        TC_STL dot_v_m(std::complex<T>(approx_res_.real(), approx_res_.imag()));
        TC_STL errC_m = dot_v_m - dot_m_std;
        TR err_re = errC_m.real()/dot_m_std.real();
        TR err_im = errC_m.imag()/dot_m_std.imag();
        errC_m = TC_STL(abs(err_re), abs(err_im));
        TC errThrust = TC( errC_m.real().get_d(), errC_m.imag().get_d() );
        return errThrust;
    }    
    T get_error_relative_T(const TC& approx_res_)
    {
        TC_STL dot_c = TC_STL(dot_m_std.real().get_d(), dot_m_std.imag().get_d() );        
        TC_STL err_d = (approx_res_ - dot_c )/abs(dot_m_std).get_d();
        
        return std::abs(err_d);
    }  

};

#endif