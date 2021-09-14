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
#ifndef __GENERATE_VECTOR_PAIR_HPP__
#define __GENERATE_VECTOR_PAIR_HPP__

#include <cstddef>
#include <utility>
#include <cmath>
#include <algorithm>
#include <random>
#include <utils/cuda_support.h>
#include <generate_vector_pair_helper.h>


/**
 * @brief      This class is a generateor of a real vector pair with desired condition number.
 *
 * @tparam     VectorOperations            { real vector operations in base type f.p. format}
 * @tparam     VectorOperationsDouble      { real vector operations in DOUBLE format}
 * @tparam     ExactDotR                   { exact dot product of real vectors }
 * @tparam     BLOCK_SIZE                  { GPU block size }
 */
template<class VectorOperations, class VectorOperationsDouble, class ExactDot, int BLOCK_SIZE = 1024>
class generate_vector_pair
{
private:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using vec_helper_t = generate_vector_pair_helper<T, T_vec>;

    std::mt19937 gen;
    T_vec x1_a_ = nullptr;
    T_vec x2_a_ = nullptr;

    double* x1_double_a_ = nullptr;
    double* x2_double_a_ = nullptr;


public:
    generate_vector_pair(VectorOperations* vec_ops_, VectorOperationsDouble* vec_ops_double_, ExactDot* exact_dot_):
    vec_ops(vec_ops_),
    vec_ops_double(vec_ops_double_),
    exact_dot(exact_dot_)
    {

        sz = vec_ops->get_vector_size();
        d1_c = new double[sz];
        d2_c = new double[sz];
        vec_helper = new vec_helper_t(sz);
        rand_mantisa_d = device_allocate<T>(sz);
        rand_exponent_d = device_allocate<T>(sz);
        
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        gen.seed(rd()); //Standard mersenne_twister_engine seeded with rd()
        vec_ops->init_vector(x1_a_); vec_ops->start_use_vector(x1_a_);
        vec_ops->init_vector(x2_a_); vec_ops->start_use_vector(x2_a_);
        
        vec_helper->init_double_vectors(x1_double_a_, x2_double_a_);

    }
    ~generate_vector_pair()
    {
        vec_helper->delete_double_vectors(x1_double_a_, x2_double_a_);   
        if( d1_c != nullptr)
        {
            delete [] d1_c;
        }
        if( d2_c != nullptr)
        {
            delete [] d2_c;
        }
        if( vec_helper != nullptr )
        {
            delete vec_helper;
        }
        if (rand_exponent_d != nullptr)
        {
            device_deallocate<T>(rand_exponent_d);
        }
        if (rand_mantisa_d != nullptr)
        {
            device_deallocate<T>(rand_mantisa_d);
        }
        vec_ops->stop_use_vector(x1_a_); vec_ops->free_vector(x1_a_); 
        vec_ops->stop_use_vector(x2_a_); vec_ops->free_vector(x2_a_);   
    }

    void dot_exact()
    {
        use_exact_dot = 1;
    }

    void dot_exact_cuda()
    {
        use_exact_dot = 2;        
    }

    T generate(T_vec& x_, T_vec& y_, T condition_number_, int use_exact_dot_cuda_ = 0)
    {
        T condition_estimate = 0.0;
        do
        {
            T b = std::log2(condition_number_);
            // we take the last X elements or a half at most
            // it is needed to adjust the vectors to the appropriate condition number
            int X = int(0.25*sz);
            int Y = 20000;//std::max(X,2000);
            std::uniform_int_distribution<> distrib(10000, Y );  //from 1000 to Y

            int size_exp = std::max( distrib(gen), int(0.5*b) ); 
            size_t N_fix_part = std::min( size_exp, int(std::round(0.5*sz)) ); 

            T b_step = 0.5*b/T(N_fix_part);
            T b_val = 0.5*b;

            T_vec x_part_c = new T[N_fix_part];
            T_vec exp_fixed = new T[N_fix_part];
            for( int j=0; j<N_fix_part; j++)
            {
                x_part_c[j]=T(std::round(b_val - T(j)*b_step));
                exp_fixed[j] = x_part_c[j];
            }

            vec_ops->assign_random(rand_exponent_d, T(0.0), b_val );
            host_2_device_cpy<T>(&rand_exponent_d[sz - N_fix_part], x_part_c, N_fix_part);
            vec_ops->assign_random(rand_mantisa_d, T(-1.0), T(1.0) );

            vec_helper->generate_C_estimated_vector(b_val, rand_mantisa_d, rand_exponent_d, x_, N_fix_part);

            vec_ops->assign_random(rand_exponent_d, T(0), b_val );
            vec_ops->assign_random(rand_mantisa_d, T(-1.0), T(1.0) );
            vec_helper->generate_C_estimated_vector(b_val, rand_mantisa_d, rand_exponent_d, y_, N_fix_part);

            device_2_host_cpy<T>(x_part_c, &x_[sz - N_fix_part], N_fix_part);
            T_vec y_part_c = new T[N_fix_part];
            device_2_host_cpy<T>(y_part_c, &rand_mantisa_d[sz - N_fix_part], N_fix_part);
            
            vec_helper->convert_vector_T_to_double(x_, x1_double_a_);
            vec_helper->convert_vector_T_to_double(y_, x2_double_a_);            
            T dot_xy = static_cast<T>(dot(x1_double_a_, x2_double_a_)); //local dot product!!!
            for( int j = 0; j<N_fix_part; j++)
            {
                //local dot product update
                vec_helper->convert_vector_T_to_double(x_, x1_double_a_);
                vec_helper->convert_vector_T_to_double(y_, x2_double_a_);
                dot_xy = static_cast<T>( dot_update(sz - N_fix_part, sz, x1_double_a_, x2_double_a_) );
                T y_l = (y_part_c[j]*std::pow<T>(T(2.0), exp_fixed[j]) - dot_xy)/x_part_c[j];
                vec_ops->set_value_at_point(y_l, sz - N_fix_part + j, y_);
            }
            // host_2_device_cpy<T>(&y_[sz - N_fix_part], y_part_c, N_fix_part);

            delete [] exp_fixed;
            delete [] y_part_c;
            delete [] x_part_c;

            condition_estimate = estimate_condition_reduction(x_, y_);
        }
        while(!isfinite(condition_estimate));

        return condition_estimate;
    }


    T condition_number_max(T_vec x1_, T_vec x2_)
    {
        return(estimate_condition_reduction(x1_, x2_));
    }

private:

    VectorOperations* vec_ops;
    VectorOperationsDouble* vec_ops_double;
    ExactDot* exact_dot;
    size_t sz;
    char use_exact_dot = 0;
    double* d1_c = nullptr;
    double* d2_c = nullptr;

    T_vec rand_exponent_d = nullptr;
    T_vec rand_mantisa_d = nullptr;

    vec_helper_t* vec_helper = nullptr;


    T estimate_condition_reduction(T_vec x1_, T_vec x2_)
    {
        
        vec_helper->convert_vector_T_to_double(x1_, x1_double_a_);
        vec_helper->convert_vector_T_to_double(x2_, x2_double_a_);
        double x1x2 = std::abs( dot(x1_double_a_, x2_double_a_) );
        vec_helper->return_abs_double_vec_inplace(x1_double_a_);
        vec_helper->return_abs_double_vec_inplace(x2_double_a_);
        double ax1aax2a = dot(x1_double_a_, x2_double_a_);
        return( T(ax1aax2a/x1x2) );
    }


    inline T dot(double* d1, double* d2)
    {
        T res = 0;
        if(use_exact_dot == 0)
        {
            res = dot_reduction(d1, d2);
        }
        else
        {
            res = dot_exact(d1, d2);
        }
        return( res );
    }

    inline double dot_reduction(double* d1, double* d2)
    {
        vec_ops_double->use_high_precision();
        double res = vec_ops_double->scalar_prod(d1, d2);
        vec_ops_double->use_standard_precision();
        return( res );
    }


    inline double dot_exact(double* d1, double* d2)
    {
        if(use_exact_dot == 1)
        {
            vec_ops_double->get(d1, d1_c);
            vec_ops_double->get(d2, d2_c);            
            exact_dot->use_cpu();
            exact_dot->set_arrays(sz, d1_c, d2_c);
            return( exact_dot->dot_exact() );
        }
        else if(use_exact_dot == 2)
        {
            vec_ops_double->get(d1, d1_c);
            vec_ops_double->get(d2, d2_c);
            exact_dot->use_gpu(sz);            
            exact_dot->set_arrays(sz, d1_c, d2_c);
            return( exact_dot->dot_exact() );            
        }
        else
        {
            return double(0.0);
        }
    }

    inline double dot_update(size_t from_, size_t to_, double* d1, double* d2)
    {
       
        T res = 0;
        if(use_exact_dot == 0)
        {
            res = dot_reduction(d1, d2);
        }
        else if(use_exact_dot == 1)
        {
            res = dot_exact(d1, d2); 
        }
        else if(use_exact_dot == 2)
        {
            vec_ops_double->get(d1, d1_c);
            vec_ops_double->get(d2, d2_c);
            exact_dot->use_gpu(sz);
            exact_dot->update_arrays(from_, to_, d1_c, d2_c);
            res = exact_dot->dot_exact();
        }
        return( res );
    }


};

#endif