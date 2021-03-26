// https://doc.lagout.org/science/0_Computer%20Science/3_Theory/Handbook%20of%20Floating%20Point%20Arithmetic.pdf

#ifndef __DOT_PRODUCT_GMP_COMPLEX_HPP__
#define __DOT_PRODUCT_GMP_COMPLEX_HPP__

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <../gmp/install/include/gmpxx.h>
#include <thrust/complex.h>

template<class T, class T_vec>
class dot_product_gmp_complex
{

using TC = thrust::complex<T>;
using TC_HP = thrust::complex<mpf_class>;

private:
    T_vec X;
    T_vec Y;
    bool array_set = false;
    unsigned int exact_prec_bits;
    size_t N;

    TC_HP dot_m;
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

        for(size_t j=0;j<N;j++)
        {
            TC_HP x_l(conj(X[j]) );
            TC_HP y_l(Y[j]);
            dot_m = dot_m + x_l*y_l;
        }
        
        thrust::complex<double> dot_exact_T = TC( dot_m.real().get_d(), dot_m.imag().get_d() );
        
        return TC(dot_exact_T);
    }
    
    void print_res()
    {
        std::cout << std::scientific << std::setprecision(128) << dot_m << std::endl;
    }
    T get_error(const TC& approx_res_)
    {
        TC_HP dot_v_m(approx_res_);
        TC_HP errC_m = dot_v_m - dot_m;
        mpf_class err_m = abs(errC_m);

        double err_d = err_m.get_d();
        return T(err_d);
    }
    T get_error_T(const TC& approx_res_)
    {
        TC dot_c = TC(dot_m.real().get_d(), dot_m.imag().get_d() );
        TC err_c = approx_res_ - dot_c;
        

        return abs(err_c);
    }
    T get_error_relative(const TC& approx_res_)
    {
        TC_HP dot_v_m(approx_res_);
        TC_HP errC_m = (dot_v_m - dot_m)/abs(dot_m);
        mpf_class err_m = abs(errC_m);

        double err_d = err_m.get_d();
        return T(err_d);
    }    
    T get_error_relative_T(const TC& approx_res_)
    {
        TC dot_c = TC(dot_m.real().get_d(), dot_m.imag().get_d() );        
        TC err_d = (approx_res_ - dot_c )/abs(dot_m).get_d();
        
        return abs(err_d);
    }  

};

#endif