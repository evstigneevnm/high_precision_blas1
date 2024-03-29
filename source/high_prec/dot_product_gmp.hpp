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

#ifndef __DOT_PRODUCT_GMP_HPP__
#define __DOT_PRODUCT_GMP_HPP__

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <gmpxx.h>
#include <high_prec/dot_product_cump.hpp>

template<class T, class T_vec>
class dot_product_gmp
{
private:
    using cump_t = dot_product_cump<T, T_vec>;
    mutable T_vec X;
    mutable T_vec Y;
    mutable bool array_set = false;
    mutable bool gpu = false;
    mutable unsigned int exact_prec_bits;
    mutable size_t N;
    cump_t* cump_class = nullptr;

    mutable mpf_class s_m;
    mutable T dot_naive_v;
    mutable T dot_ogita_v;
    mutable T dot_fma_v;


public:
    dot_product_gmp(unsigned int exact_prec_bits_ = 512):
    exact_prec_bits(exact_prec_bits_)
    {
        mpf_set_default_prec(exact_prec_bits);
    }
    
    ~dot_product_gmp()
    {
        if( cump_class != nullptr)
        {
            delete cump_class;
        }
    }

    void use_gpu(size_t sz_)
    {
        if(cump_class == nullptr)
        {
            cump_class = new cump_t(sz_, exact_prec_bits);
        }
        gpu = true;

    }

    void use_cpu()
    {
        gpu = false;
    }

    void set_arrays(size_t N_, T_vec input_array_1_, T_vec input_array_2_) const
    {
        if(gpu)
        {
            cump_class->set_arrays(input_array_1_, input_array_2_);
        }
        X = input_array_1_;
        Y = input_array_2_;
        N = N_;
        array_set = true;

    }
    
    void update_arrays(size_t from_, size_t to_, T_vec& input_array_1_, T_vec& input_array_2_)
    {
        if(array_set&&gpu)
        {
            cump_class->update_arrays(from_, to_, input_array_1_, input_array_2_);
        }
    }

    T dot_exact() const
    {

        
        s_m = mpf_class(0, exact_prec_bits);
        
        if(gpu)
        {
            cump_class->dot_exact(s_m);
        }
        else
        {
            for(size_t j=0;j<N;j++)
            {
                mpf_class x_l(X[j], exact_prec_bits);
                mpf_class y_l(Y[j], exact_prec_bits);
                s_m = s_m + x_l*y_l;
            }
        }        
        
        double dot_exact_T = s_m.get_d();
        return T(dot_exact_T);
    }
    T sum_two(const T one_, const T two)
    {
        mpf_class res_mpf = mpf_class(one_, exact_prec_bits)+ mpf_class(two, exact_prec_bits);
        return( T( res_mpf.get_d() ) );
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