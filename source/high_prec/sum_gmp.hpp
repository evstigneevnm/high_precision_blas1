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

// https://doc.lagout.org/science/0_Computer%20Science/3_Theory/Handbook%20of%20Floating%20Point%20Arithmetic.pdf

#ifndef __SUM_GMP_HPP__
#define __SUM_GMP_HPP__

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <../gmp/install/include/gmpxx.h>


template<class T, class T_vec>
class sum_gmp
{
private:
    T_vec X;
    T_vec Y;
    bool array_set = false;
    unsigned int exact_prec_bits;
    size_t N;

    mpf_class s_m;
    T dot_naive_v;
    T dot_ogita_v;
    T dot_fma_v;


public:
    sum_gmp(unsigned int exact_prec_bits_ = 512, bool use_gpu_ = false):
    exact_prec_bits(exact_prec_bits_)
    {
        mpf_set_default_prec(exact_prec_bits);
    }
    
    ~sum_gmp()
    {
    }

    void set_array(size_t N_, T_vec& input_array_1_)
    {

        X = input_array_1_;

        N = N_;
        array_set = true;
    }
    
    T sum_exact()
    {

        s_m = mpf_class(0, exact_prec_bits);
        for(size_t j=0;j<N;j++)
        {
            mpf_class x_l(X[j], exact_prec_bits);
            s_m = s_m + x_l;
        }
        
        double sum_exact_T = s_m.get_d();
        
        return T(sum_exact_T);
    }
    T asum_exact()
    {

        s_m = mpf_class(0, exact_prec_bits);
        for(size_t j=0;j<N;j++)
        {
            mpf_class x_l( std::abs(X[j]), exact_prec_bits);
            s_m = s_m + x_l;
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