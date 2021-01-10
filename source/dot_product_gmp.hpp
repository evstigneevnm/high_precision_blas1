// https://doc.lagout.org/science/0_Computer%20Science/3_Theory/Handbook%20of%20Floating%20Point%20Arithmetic.pdf

#ifndef __DOT_PRODUCT_HPP_
#define __DOT_PRODUCT_HPP_

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <../gmp/install/include/gmpxx.h>


template<class T>
class dot_product_gmp
{
private:
    T *X, *Y;
    bool array_set = false;
    unsigned int exact_prec_bits;
    size_t N;

    mpf_class s_m;
    T dot_naive_v;
    T dot_ogita_v;
    T dot_fma_v;


    T two_prod(T &t, T a, T b) // [1], pdf: 71, 169, 198, 
    {
        T p = a*b;
        t = std::fma(a, b, -p);
        return p;
    }

    T two_sum (T &t, T a, T b) 
    {
        T s = a+b;
        T bs = s-a;
        T as = s-bs;
        t = (b-bs) + (a-as);
        return s;
    }

public:
    dot_product_gmp(unsigned int exact_prec_bits_ = 512):
    exact_prec_bits(exact_prec_bits_)
    {
        mpf_set_default_prec(exact_prec_bits);
    }
    
    ~dot_product_gmp()
    {
    }

    void set_arrays(size_t N_, T*& input_array_1_, T*& input_array_2_)
    {

        X = input_array_1_;
        Y = input_array_2_;

        N = N_;
        array_set = true;
    }
    
    T dot_exact()
    {

        s_m = mpf_class(0, exact_prec_bits);
        for(size_t j=0;j<N;j++)
        {
            mpf_class x_l(X[j], exact_prec_bits);
            mpf_class y_l(Y[j], exact_prec_bits);
            s_m = s_m + x_l*y_l;
        }
        
        double dot_exact_T = s_m.get_d();
        
        return T(dot_exact_T);
    }
    

    T get_error(const T& approx_res_)
    {
        mpf_class dot_v_m(approx_res_, exact_prec_bits);
        mpf_class err_m = dot_v_m - s_m;
        
        double err_d = err_m.get_d();
        return std::abs(T(err_d));
    }
};

#endif