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
#ifndef __SUMS_HPP__
#define __SUMS_HPP__


#include <cstddef>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <../gmp/install/include/gmpxx.h>



template<class T>
class sums
{
private:
    T* input_array;
    size_t N;
    bool use_gpu;
    bool array_set = false;
    mpf_class sum_m;
    T sum_naive_val = 0;
    T sum_kahan_val = 0;
    T sum_rump_val = 0;
    unsigned int exact_prec_bits;

    T fast_two_sum (T &t, T a, T b)
    {
        T s = a+b;
        t = b-(s-a);
        return s;
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
    sums(unsigned int exact_prec_bits_ = 512, bool use_gpu_ = false):
    exact_prec_bits(exact_prec_bits_),
    use_gpu(use_gpu_)
    {
        mpf_set_default_prec(exact_prec_bits);
    }

    ~sums()
    {

    }



    void set_array(size_t N_, T*& input_array_)
    {
        if(use_gpu)
        {

        }
        input_array = input_array_;

        N = N_;
        array_set = true;
    }

    T sum_exact()
    {

        sum_m = mpf_class(0, exact_prec_bits);
        for(size_t j=0;j<N;j++)
        {
            sum_m = sum_m + input_array[j];
        }
        
        double sum_exact_T = sum_m.get_d();
        
        return T(sum_exact_T);
    }

    T sum_naive()
    {
        
        T sum = T(0.0);
        for(size_t j=0; j<N; j++)
        {
            sum += input_array[j];
        }
        sum_naive_val = sum;
        return sum;

    }
    T error_naive()
    {
        mpf_class sum_naive_val_m(sum_naive_val, exact_prec_bits);
        mpf_class err_m = sum_naive_val_m - sum_m;
        double err_d = err_m.get_d();
        return T(err_d);
    }

    void error_check()
    {
        T s_d = std::sqrt(2000.0);
        mpf_class s_m(s_d, exact_prec_bits); 
        mpf_class v_m(2000.0, exact_prec_bits);
        mpf_class sqrt_m = sqrt(v_m);
        mpf_class err_m = s_m - s_d;
        mpf_class err_mm = s_m - sqrt_m;
        printf("err_d = %.60le\n", err_m.get_d() );
        printf("err_m = %.60le\n", err_mm.get_d() );

    }

    T sum_kahan()
    {
        T sum = T(0.0), c = T(0.0);
        for(size_t j=0; j<N; j++)
        {
            T x = input_array[j];
            T y = x + c;
            sum = fast_two_sum(c, sum, y);
        }
        sum_kahan_val = sum;
        return sum;
    }
    T error_kahan()
    {
        mpf_class sum_kahan_val_m(sum_kahan_val, exact_prec_bits);
        mpf_class err_m = sum_kahan_val_m - sum_m;
        double err_d = err_m.get_d();
        return T(err_d);
    }

    T sum_rump() 
    {
        T s = T(0.0), c = T(0.0), e = T(0.0);

        for(size_t j=0; j<N; j++)
        {
            T x = input_array[j];        
            s = two_sum(e, s, x);
            c += e;
        }
        T sum = s + c;
        sum_rump_val = sum;
        return sum;
    }
    
    T error_rump()
    {
        mpf_class sum_rump_val_m(sum_rump_val, exact_prec_bits);
        mpf_class err_m = sum_rump_val_m - sum_m;
        double err_d = err_m.get_d();
        return T(err_d);
    }



};

#endif