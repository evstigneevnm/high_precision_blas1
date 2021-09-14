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
#ifndef __GENERATE_VECTOR_PAIR_COMPLEX_HPP__
#define __GENERATE_VECTOR_PAIR_COMPLEX_HPP__

#include <cstddef>
#include <utility>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <utils/cuda_support.h>
#include <generate_vector_pair_helper_complex.h>


/**
 * @brief      This class is a generateor of a complex vector pair with desired condition number.
 *
 * @tparam     VectorOperations        { complex vector operations }
 * @tparam     VectorOperationsR       { real vector operations in base type }
 * @tparam     VectorOperationsDoubleR { real vector operations in DOUBLE f.p. }
 * @tparam     ExactDotR               { exact dot product of REAL vectors }
 * @tparam     BLOCK_SIZE              { GPU block size }
 */
template<class VectorOperations, class VectorOperationsR, class VectorOperationsDoubleR, class ExactDotR, int BLOCK_SIZE = 1024>
class generate_vector_pair_complex
{
private:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using TR = typename VectorOperationsR::scalar_type;
    using TR_vec = typename VectorOperationsR::vector_type;
    using vec_helper_t = generate_vector_pair_helper_complex<T, T_vec>;


    std::mt19937 gen;
    T_vec x1_a_ = nullptr;
    T_vec x2_a_ = nullptr;

    double* x1_double_a_ = nullptr;
    double* x2_double_a_ = nullptr;

    double* x1_double_r_ = nullptr;
    double* x2_double_r_ = nullptr;
    double* x1_double_i_ = nullptr;
    double* x2_double_i_ = nullptr;

    double* r1_c = nullptr;
    double* r2_c = nullptr;
public:
    generate_vector_pair_complex(VectorOperations* vec_ops_, VectorOperationsR* vec_opsR_, VectorOperationsDoubleR* vec_ops_double_R_, ExactDotR* exact_dot_):
    vec_ops(vec_ops_),
    vec_opsR(vec_opsR_),
    vec_ops_double_R(vec_ops_double_R_),
    exact_dot(exact_dot_)
    {
        use_exact_dot = 0;
        sz = vec_ops->get_vector_size();
        d1_c = new T[sz];
        d2_c = new T[sz];
        vec_helper = new vec_helper_t(sz);
        vec_ops->init_vector(rand_mantisa_d); vec_ops->start_use_vector(rand_mantisa_d);
        vec_opsR->init_vector(rand_exponent_d); vec_opsR->start_use_vector(rand_exponent_d);
        r1_c = new double[sz];
        r2_c = new double[sz];

        
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        gen.seed(rd()); //Standard mersenne_twister_engine seeded with rd()
        vec_ops->init_vector(x1_a_); vec_ops->start_use_vector(x1_a_);
        vec_ops->init_vector(x2_a_); vec_ops->start_use_vector(x2_a_);
        
        vec_helper->init_double_vectors(x1_double_a_, x2_double_a_);
        vec_helper->init_double_vectors(x1_double_i_, x2_double_i_);
        vec_helper->init_double_vectors(x1_double_r_, x2_double_r_);

    }
    ~generate_vector_pair_complex()
    {
        vec_helper->delete_double_vectors(x1_double_r_, x2_double_r_);
        vec_helper->delete_double_vectors(x1_double_i_, x2_double_i_);
        vec_helper->delete_double_vectors(x1_double_a_, x2_double_a_);   
        
        if( r1_c != nullptr)
        {
            delete [] r1_c;
        } 
        if( r2_c != nullptr)
        {
            delete [] r2_c;
        }  
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
        vec_ops->stop_use_vector(rand_mantisa_d); vec_ops->free_vector(rand_mantisa_d);
        vec_opsR->stop_use_vector(rand_exponent_d); vec_opsR->free_vector(rand_exponent_d);

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
    

    std::pair<TR,TR> generate_pure_im(T_vec& x_, T_vec& y_, TR condition_number_, TR_vec& xi_, TR_vec& yi_, std::string part_of_dot_product = "IR", int use_exact_dot_cuda_ = 0)
    {
        std::pair<TR,TR> condition_estimate = {0,0};
        
        std::uniform_int_distribution<> test_re_im(0, 1 );

        do{
            TR b = std::log2(condition_number_);
            // we take the last X elements or a half at most
            // it is needed to adjust the vectors to the appropriate condition number
            int X = int(0.25*sz);
            int Y = 20000;//std::max(X,2000);
            std::uniform_int_distribution<> distrib(10000, Y );  //from 1000 to Y

            int size_exp = std::max( distrib(gen), int(0.5*b) ); 
            size_t N_fix_part = std::min( size_exp, int(std::round(0.5*sz)) ); 

            TR b_step = 0.5*b/TR(N_fix_part);
            TR b_val = 0.5*b;

            T_vec x_part_c = new T[N_fix_part];
            TR_vec exp_fixed = new TR[N_fix_part];
            for( int j=0; j<N_fix_part; j++)
            {
                if(part_of_dot_product == "RI")
                {
                    x_part_c[j]=T(std::round(b_val - TR(j)*b_step), 0.0);
                    exp_fixed[j] = x_part_c[j].real();
                }
                if(part_of_dot_product == "IR")
                {
                    x_part_c[j]=T(0.0, std::round(b_val - TR(j)*b_step));
                    exp_fixed[j] = x_part_c[j].imag();
                }
            }

            vec_opsR->assign_random(rand_exponent_d, TR(0.0), TR(b_val) );
            host_2_device_cpy<TR>(&rand_exponent_d[sz - N_fix_part], exp_fixed, N_fix_part);
            if(part_of_dot_product == "RI")
            {
                vec_ops->assign_random(rand_mantisa_d, T(-1.0,0.0), T(1.0,0.0) );
            }
            if(part_of_dot_product == "IR")
            {
                vec_ops->assign_random(rand_mantisa_d, T(0.0,-1.0), T(0.0,1.0) );
            }

            vec_helper->generate_C_estimated_vector(b_val, rand_mantisa_d, rand_exponent_d, x_, N_fix_part);

            vec_opsR->assign_random(rand_exponent_d, TR(0.0), TR(b_val) );
            if(part_of_dot_product == "RI")
            {
                vec_ops->assign_random(rand_mantisa_d, T(0.0,-1.0), T(0.0,1.0) );
            }
            if(part_of_dot_product == "IR")
            {
                vec_ops->assign_random(rand_mantisa_d, T(-1.0,0), T(1.0,0) );
            }
            vec_helper->generate_C_estimated_vector(b_val, rand_mantisa_d, rand_exponent_d, y_, N_fix_part);

            device_2_host_cpy<T>(x_part_c, &x_[sz - N_fix_part], N_fix_part);
            T_vec y_part_c = new T[N_fix_part];
            device_2_host_cpy<T>(y_part_c, &rand_mantisa_d[sz - N_fix_part], N_fix_part);

            for( int j = 0; j<N_fix_part; j++)
            {             

                vec_helper->split_complex_vector_to_reals(x_, x1_double_r_, x1_double_i_);
                vec_helper->split_complex_vector_to_reals(y_, x2_double_r_, x2_double_i_);
                TR y_l_real = 0;
                TR y_l_imag = 0;
                
                if(part_of_dot_product == "RI")
                {
                    TR RI = dot_real(x1_double_r_, x2_double_i_);
                    y_l_imag = ( y_part_c[j].real()*std::pow<TR>(TR(2.0), exp_fixed[j]) - RI)/x_part_c[j].real();
                }
                if(part_of_dot_product == "IR")
                {
                    TR IR = dot_real(x1_double_i_, x2_double_r_);
                    y_l_real = (y_part_c[j].imag()*std::pow<TR>(TR(2.0), exp_fixed[j]) - IR)/x_part_c[j].imag(); 
                }
      

                vec_ops->set_value_at_point(T(y_l_real, y_l_imag), sz - N_fix_part + j, y_);

            }

            delete [] exp_fixed;
            delete [] y_part_c;
            delete [] x_part_c;

            condition_estimate = estimate_condition_reduction(x_, y_);
            condition_estimate.first = 0;

            vec_helper->split_complex_vector_to_reals(x_, xi_, x1_double_i_);
            vec_helper->split_complex_vector_to_reals(y_, yi_, x2_double_i_);

        }
        while(!(isfinite(condition_estimate.first)||isfinite(condition_estimate.second)) );

        return condition_estimate;
    }


    std::pair<TR,TR> generate_pure_re(T_vec& x_, T_vec& y_, TR condition_number_, TR_vec& xr_, TR_vec& yr_, std::string part_of_dot_product = "RR", int use_exact_dot_cuda_ = 0)
    {
        std::pair<TR,TR> condition_estimate = {0,0};
        
        std::uniform_int_distribution<> test_re_im(0, 1 );

        do{
            TR b = std::log2(condition_number_);
            // we take the last X elements or a half at most
            // it is needed to adjust the vectors to the appropriate condition number
            int X = int(0.25*sz);
            int Y = 20000;//std::max(X,2000);
            std::uniform_int_distribution<> distrib(10000, Y );  //from 1000 to Y

            int size_exp = std::max( distrib(gen), int(0.5*b) ); 
            size_t N_fix_part = std::min( size_exp, int(std::round(0.5*sz)) ); 

            TR b_step = 0.5*b/TR(N_fix_part);
            TR b_val = 0.5*b;

            T_vec x_part_c = new T[N_fix_part];
            TR_vec exp_fixed = new TR[N_fix_part];
            for( int j=0; j<N_fix_part; j++)
            {
                if(part_of_dot_product == "RR")
                {
                    x_part_c[j]=T(std::round(b_val - TR(j)*b_step), 0.0);
                    exp_fixed[j] = x_part_c[j].real();
                }
                else
                {
                    x_part_c[j]=T(0.0, std::round(b_val - TR(j)*b_step));
                    exp_fixed[j] = x_part_c[j].imag();
                }
                
            }

            vec_opsR->assign_random(rand_exponent_d, TR(0.0), TR(b_val) );
            host_2_device_cpy<TR>(&rand_exponent_d[sz - N_fix_part], exp_fixed, N_fix_part);
            if(part_of_dot_product == "RR")
            {
                vec_ops->assign_random(rand_mantisa_d, T(-1.0,0.0), T(1.0,0.0) );
            }
            if(part_of_dot_product == "II")
            {
                vec_ops->assign_random(rand_mantisa_d, T(0.0,-1.0), T(0.0,1.0) );
            }

            vec_helper->generate_C_estimated_vector(b_val, rand_mantisa_d, rand_exponent_d, x_, N_fix_part);

            vec_opsR->assign_random(rand_exponent_d, TR(0.0), TR(b_val) );

            if(part_of_dot_product == "RR")
            {
                vec_ops->assign_random(rand_mantisa_d, T(-1.0,0.0), T(1.0,0.0) );
            }
            if(part_of_dot_product == "II")
            {
                vec_ops->assign_random(rand_mantisa_d, T(0.0,-1.0), T(0.0,1.0) );
            }

            vec_helper->generate_C_estimated_vector(b_val, rand_mantisa_d, rand_exponent_d, y_, N_fix_part);

            device_2_host_cpy<T>(x_part_c, &x_[sz - N_fix_part], N_fix_part);
            T_vec y_part_c = new T[N_fix_part];
            device_2_host_cpy<T>(y_part_c, &rand_mantisa_d[sz - N_fix_part], N_fix_part);

            for( int j = 0; j<N_fix_part; j++)
            {             

                vec_helper->split_complex_vector_to_reals(x_, x1_double_r_, x1_double_i_);
                vec_helper->split_complex_vector_to_reals(y_, x2_double_r_, x2_double_i_);
                TR y_l_real = 0;
                TR y_l_imag = 0;

                
                
                if(part_of_dot_product == "RR")
                {
                    TR RR = dot_real(x1_double_r_, x2_double_r_);
                    y_l_real = ( y_part_c[j].real()*std::pow<TR>(TR(2.0), exp_fixed[j]) - RR)/x_part_c[j].real();
                }
                else
                {
                    TR II = dot_real(x1_double_i_, x2_double_i_);
                    y_l_imag = (y_part_c[j].imag()*std::pow<TR>(TR(2.0), exp_fixed[j]) - II)/x_part_c[j].imag();
                }

                vec_ops->set_value_at_point(T(y_l_real, y_l_imag), sz - N_fix_part + j, y_);


            }

            

            delete [] exp_fixed;
            delete [] y_part_c;
            delete [] x_part_c;


            condition_estimate = estimate_condition_reduction(x_, y_);
            

            vec_helper->split_complex_vector_to_reals(x_, xr_, x1_double_i_);
            vec_helper->split_complex_vector_to_reals(y_, yr_, x2_double_i_);

        }
        while(!(isfinite(condition_estimate.first)||isfinite(condition_estimate.second)) );


        return condition_estimate;
    }


    std::pair<TR,TR> generate(T_vec& x_, T_vec& y_, TR condition_number_, int use_exact_dot_cuda_ = 0)
    {
        std::pair<TR,TR> condition_estimate = {0,0};
        
        std::uniform_int_distribution<> test_re_im(0, 1 );
        int gen_re_or_im = test_re_im(gen);

        int Y = 20000;//std::max(X,2000);
        T_vec x_part_c = new T[Y];
        TR_vec exp_fixed = new TR[Y];
        T_vec y_part_c = new T[Y];
        bool repeat = true;
        do{
            TR b = std::log2(condition_number_);
            // we take the last X elements or a half at most
            // it is needed to adjust the vectors to the appropriate condition number
            int X = int(0.25*sz);
            std::uniform_int_distribution<> distrib(10000, Y );  //from 1000 to Y

            int size_exp = std::max( distrib(gen), int(0.5*b) ); 
            size_t N_fix_part = std::min( size_exp, int(std::round(0.5*sz)) ); 

            TR b_step = 0.5*b/TR(N_fix_part);
            TR b_val = 0.5*b;


            for( int j=0; j<N_fix_part; j++)
            {
                x_part_c[j]=T(std::round(b_val - TR(j)*b_step), std::round(b_val - TR(j)*b_step));
                exp_fixed[j] = x_part_c[j].real();
            }

            vec_opsR->assign_random(rand_exponent_d, TR(0.0), TR(b_val) );
            host_2_device_cpy<TR>(&rand_exponent_d[sz - N_fix_part], exp_fixed, N_fix_part);
            vec_ops->assign_random(rand_mantisa_d, T(-1.0,-1.0), T(1.0,1.0) );

            vec_helper->generate_C_estimated_vector(b_val, rand_mantisa_d, rand_exponent_d, x_, N_fix_part);

            vec_opsR->assign_random(rand_exponent_d, TR(0.0), TR(b_val) );
            vec_ops->assign_random(rand_mantisa_d, T(-1.0,-1.0), T(1.0,1.0) );
            vec_helper->generate_C_estimated_vector(b_val, rand_mantisa_d, rand_exponent_d, y_, N_fix_part);

            device_2_host_cpy<T>(x_part_c, &x_[sz - N_fix_part], N_fix_part);

            device_2_host_cpy<T>(y_part_c, &rand_mantisa_d[sz - N_fix_part], N_fix_part);

            for( int j = 0; j<N_fix_part; j++)
            {             


                vec_helper->split_complex_vector_to_reals(x_, x1_double_r_, x1_double_i_);
                vec_helper->split_complex_vector_to_reals(y_, x2_double_r_, x2_double_i_);
                TR y_l_real = 0;
                TR y_l_imag = 0;
                if(gen_re_or_im==0)
                {

                    TR RR = dot_real(x1_double_r_, x2_double_r_);
                    TR II = dot_real(x1_double_i_, x2_double_i_);
                    y_l_real = ( y_part_c[j].real()*std::pow<TR>(TR(2.0), exp_fixed[j]) - RR)/x_part_c[j].real();
                    y_l_imag = (y_part_c[j].imag()*std::pow<TR>(TR(2.0), exp_fixed[j]) - II)/x_part_c[j].imag();
                
                }
                else
                {
                    TR RI = dot_real(x1_double_r_, x2_double_i_);
                    TR IR = dot_real(x1_double_i_, x2_double_r_);
                    y_l_imag = ( y_part_c[j].real()*std::pow<TR>(TR(2.0), exp_fixed[j]) - RI)/x_part_c[j].real();
                    y_l_real = (y_part_c[j].imag()*std::pow<TR>(TR(2.0), exp_fixed[j]) - IR)/x_part_c[j].imag();                    
                }
                vec_ops->set_value_at_point(T(y_l_real, y_l_imag), sz - N_fix_part + j, y_);


            }


            condition_estimate = estimate_condition_reduction(x_, y_);

            bool is_finite1 = isfinite(condition_estimate.first);
            bool is_finite2 = isfinite(condition_estimate.second);
            bool size1 = condition_estimate.first > 1.0*sz;
            bool size2 = condition_estimate.second > 1.0*sz;
            TR condition_estimate_component = condition_estimate.first;// - condition_number_;
            if(gen_re_or_im == 1)
            {
                condition_estimate_component = condition_estimate.second;// - condition_number_;
            }

            bool failed = ((std::abs<TR>(std::abs(std::log(condition_number_) - std::log(condition_estimate_component))))>30.0);
            repeat = (!(is_finite1||is_finite2))||failed;

            // std::cout << condition_number_ << " " << condition_estimate_component << " " << std::abs(std::log(condition_number_) - std::log(condition_estimate_component)) << std::endl;
        }
        while( repeat );
        
        delete [] exp_fixed;
        delete [] y_part_c;
        delete [] x_part_c;        

        return condition_estimate;
    }



    T get_dot_of_abs_vectors()
    {
        return(abs_dot);
    }


    TR condition_number_max(T_vec x1_, T_vec x2_) const
    {
        auto cn = estimate_condition_reduction(x1_, x2_);
        return(std::max(cn.first, cn.second));
    }

private:

    VectorOperations* vec_ops;
    VectorOperationsR* vec_opsR;
    VectorOperationsDoubleR* vec_ops_double_R;
    ExactDotR* exact_dot;

    size_t sz;
    char use_exact_dot = 0;
    T_vec d1_c = nullptr;
    T_vec d2_c = nullptr;

    TR_vec rand_exponent_d = nullptr;
    T_vec rand_mantisa_d = nullptr;

    vec_helper_t* vec_helper = nullptr;

    mutable T abs_dot = T(0,0);

    std::pair<TR,TR> estimate_condition_reduction(T_vec x1_, T_vec x2_) const
    {
        
        
        vec_helper->split_complex_vector_to_reals(x1_, x1_double_r_, x1_double_i_);
        vec_helper->split_complex_vector_to_reals(x2_, x2_double_r_, x2_double_i_);
        T res_dot = dot(x1_double_r_, x2_double_r_, x1_double_i_, x2_double_i_);
        T res_abs_dot = T(std::abs(res_dot.real()), std::abs(res_dot.imag()));

        vec_helper->return_abs_4double_vec_inplace(x1_double_r_, x2_double_r_, x1_double_i_, x2_double_i_);
        TR adot1 = dot_real(x1_double_r_, x2_double_r_);
        TR adot2 = dot_real(x1_double_i_, x2_double_i_);
        TR adot3 = dot_real(x1_double_r_, x2_double_i_);
        TR adot4 = dot_real(x1_double_i_, x2_double_r_);

        TR cond_re = 0;
        TR cond_im = 0;
        abs_dot = T(adot1 + adot2, adot3 + adot4);

        if(std::abs(res_abs_dot.real())>0.0)
            cond_re = ( abs_dot.real() )/res_abs_dot.real();
        if(std::abs(res_abs_dot.imag())>0.0)
            cond_im = ( abs_dot.imag() )/res_abs_dot.imag();
        

        return( std::pair<TR,TR>(cond_re, cond_im) );
    }

    T dot(double* x1_r, double* x2_r, double* x1_i, double* x2_i) const
    {
        TR RR = dot_real(x1_r, x2_r);
        TR II = dot_real(x1_i, x2_i);
        
        TR RI = dot_real(x1_r, x2_i);
        TR IR = dot_real(x1_i, x2_r);

        TR dot_real = exact_dot->sum_two(RR, II);
        TR dot_imag = exact_dot->sum_two(RI,-IR);

        return( T(dot_real, dot_imag) );
    }



    TR dot_real(double* d1, double* d2) const
    {
        TR res = 0;
        if(use_exact_dot == 1)
        {

            device_2_host_cpy<double>(r1_c , d1, sz );
            device_2_host_cpy<double>(r2_c , d2, sz );
            exact_dot->set_arrays(sz, r1_c, r2_c);
            res = static_cast<TR>( exact_dot->dot_exact() );
        }
        else
        {
            vec_ops_double_R->use_high_precision();
            res = static_cast<TR>( vec_ops_double_R->scalar_prod(d1, d2) );
            vec_ops_double_R->use_standard_precision();
        }
        return(res);
    }



};

#endif