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
#ifndef __DOT_PRODUCT_CUMP_HPP__
#define __DOT_PRODUCT_CUMP_HPP__

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <gmpxx.h>
#include <cump.h>
#include <high_prec/cump_blas_kernels.h>

template<class T, class T_vec>
class dot_product_cump
{
private:
    cump_blas_kernels<>* cump_blas;

public:
    dot_product_cump(size_t sz_, unsigned int prec_):
    sz(sz_),
    prec(prec_)
    {
        cumpf_set_default_prec (prec);
        cump_blas = new cump_blas_kernels<>(sz, prec);
        X = new mpf_t[sz_];
        Y = new mpf_t[sz_];
        
        #pragma omp parallel for
        for(int j = 0; j < sz; j++)
        {
            mpf_init2(X[j],prec);
            mpf_init2(Y[j],prec);
        }


    }
    ~dot_product_cump()
    {
        if(X != nullptr)
        {
            for(int j = 0; j < sz; j++)
            {
                mpf_clear(X[j]);           
            }
            delete [] X;
        }
        
        if(Y != nullptr)
        {
            for(int j = 0; j < sz; j++)
            {
                mpf_clear(Y[j]);           
            }            
            delete [] Y;
        }

        if(cump_blas!=nullptr)
        {
            delete cump_blas;
        }
    }
    
    void set_arrays(T_vec& input_array_1_, T_vec& input_array_2_)
    {

        #pragma omp parallel for
        for( int i = 0; i < sz; i++ )
        {
            mpf_set_d(X[i], static_cast<double>(input_array_1_[i]) );
            mpf_set_d(Y[i], static_cast<double>(input_array_2_[i]) );
        }

    }   

    void update_arrays(size_t from_, size_t to_, T_vec& input_array_1_, T_vec& input_array_2_)
    {
        if(to_ <= sz)
        {
            #pragma omp parallel for
            for(int j=from_; j<to_; j++)
            {
                mpf_set_d(X[j], static_cast<double>(input_array_1_[j]) );
                mpf_set_d(Y[j], static_cast<double>(input_array_2_[j]) );            
            }
        }
    }


    T dot_benchmark(int repeats)
    {
        mpf_class s_m = mpf_class(0, prec);
        cump_blas->use_benchmark(repeats);
        dot_exact(s_m);
        double dot_T = s_m.get_d();
        return T(dot_T);
    }
    std::vector<float> get_repeated_execution_time_milliseconds()
    {
        return( cump_blas->get_repeated_execution_time_milliseconds() );
    }

    T dot()
    {
        mpf_class s_m = mpf_class(0, prec);
        dot_exact(s_m);
        double dot_T = s_m.get_d();
        return T(dot_T);
    }

    void dot_exact(mpf_class& res_)
    {
        cumpf_array_init_set_mpf (X_, X, sz);
        cumpf_array_init_set_mpf (Y_, Y, sz);
        cumpf_array_t device_result;
        cumpf_array_init2(device_result, 1, prec);
        
        cump_blas->dot(X_, Y_, device_result);

        mpf_t host_result;
        mpf_init2(host_result, prec);
        mpf_array_set_cumpf(&host_result, device_result, 1);
        mpf_set(res_.get_mpf_t(), host_result);
        cumpf_array_clear (device_result);
        cumpf_array_clear (X_);
        cumpf_array_clear (Y_);

    }

    float get_execution_time_milliseconds()
    {
        return( cump_blas->get_execution_time_milliseconds() );
    }

private:
    mpf_t* X = nullptr;
    mpf_t* Y = nullptr;
    cumpf_array_t X_, Y_;
    size_t sz;
    unsigned int prec;

};


#endif