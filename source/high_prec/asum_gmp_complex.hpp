// https://doc.lagout.org/science/0_Computer%20Science/3_Theory/Handbook%20of%20Floating%20Point%20Arithmetic.pdf

#ifndef __ASUM_GMP_COMPLEX_HPP__
#define __ASUM_GMP_COMPLEX_HPP__

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <../gmp/install/include/gmpxx.h>
#include <thrust/complex.h>

template<class T, class T_vec>
class asum_gmp
{
using TC = thrust::complex<T>;
using TC_HP = thrust::complex<mpf_class>;

private:
    T_vec X;
    T_vec Y;
    bool array_set = false;
    unsigned int exact_prec_bits;
    size_t N;

    mpf_class s_m;



public:
    asum_gmp(unsigned int exact_prec_bits_ = 512):
    exact_prec_bits(exact_prec_bits_)
    {
        mpf_set_default_prec(exact_prec_bits);
    }
    
    ~asum_gmp()
    {
    }

    void set_array(size_t N_, T_vec& input_array_1_)
    {

        X = input_array_1_;

        N = N_;
        array_set = true;
    }
    
    T asum_exact()
    {

        s_m = mpf_class(0, exact_prec_bits);
        mpf_class s_l = mpf_class(0, exact_prec_bits);
        for(size_t j=0;j<N;j++)
        {
            TC_HP x_l(X[j]);
            s_l = abs(x_l.real()) + abs(x_l.imag());
            s_m = s_m + s_l;
        }
        
        double sum_exact_T = s_m.get_d();
        
        return T(sum_exact_T);
    }
    
    void print_res()
    {
        std::cout << std::scientific << std::setprecision(128) << s_m << std::endl;
    }

    T get_error(const T& approx_res_)
    {
        mpf_class dot_v_m(approx_res_, exact_prec_bits);
        mpf_class err_m = dot_v_m - s_m;
        
        double err_d = err_m.get_d();
        return std::abs(T(err_d));
    }
    T get_error_T(const T& approx_res_)
    {

        double err_d = approx_res_ - s_m.get_d();
        
        return std::abs(T(err_d));
    }
    T get_error_relative(const T& approx_res_)
    {
        mpf_class dot_v_m(approx_res_, exact_prec_bits);
        mpf_class err_m = (dot_v_m - s_m)/s_m;
        
        double err_d = err_m.get_d();
        return std::abs(T(err_d));
    }    
    T get_error_relative_T(const T& approx_res_)
    {
        double err_d = (approx_res_ - s_m.get_d())/s_m.get_d();
        
        return std::abs(T(err_d));
    }      
};

#endif