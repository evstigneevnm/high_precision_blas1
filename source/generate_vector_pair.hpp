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
 * @brief      This class describes a generator that returns two vector pairs with the prescribed condition number.
 *
 * @tparam     VectorOperations      { class that implements vector operations }
 * @tparam     ExactDot              { class that implement exact dot product }
 * @tparam     ReductionClass        { calss that implements reduction }
 * @tparam     ReductionClassDouble  { class that implements reduction of the double precision}
 * @tparam     BLOCK_SIZE            { optional parameter that presents size of the block for GPUs }
 */
template<class VectorOperations, class ExactDot, class ReductionClass, class ReductionClassDouble, int BLOCK_SIZE = 1024>
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
    generate_vector_pair(VectorOperations* vec_ops_, ExactDot* exact_dot_, ReductionClass* reduction_, ReductionClassDouble* reduction_double_):
    vec_ops(vec_ops_),
    exact_dot(exact_dot_),
    reduction(reduction_),
    reduction_double(reduction_double_)
    {

        sz = vec_ops->get_vector_size();
        d1_c = new T[sz];
        d2_c = new T[sz];
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

    void use_exact_dot_cond()
    {
        use_exact_dot = 1;
    }

    void use_exact_dot_cond_cuda()
    {
        use_exact_dot = 2;
    }

    T generate(T_vec& x_, T_vec& y_, T condition_number_, int use_exact_dot_cuda_ = 0)
    {
        T condition_estimate = 0.0;
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

            for( int j = 0; j<N_fix_part; j++)
            {
                
                T dot_xy = dot(x_, y_); //local dot product!!!
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



private:

    VectorOperations* vec_ops;
    ExactDot* exact_dot;
    ReductionClass* reduction;
    ReductionClassDouble* reduction_double;
    size_t sz;
    char use_exact_dot = 0;
    T_vec d1_c = nullptr;
    T_vec d2_c = nullptr;

    T_vec rand_exponent_d = nullptr;
    T_vec rand_mantisa_d = nullptr;

    vec_helper_t* vec_helper = nullptr;


    T estimate_condition_reduction(T_vec x1_, T_vec x2_)
    {
        
        
        vec_helper->convert_vector_T_to_double(x1_, x1_double_a_);
        vec_helper->convert_vector_T_to_double(x2_, x2_double_a_);
        double x1x2 = std::abs( reduction_double->dot(x1_double_a_, x2_double_a_) );
        vec_helper->return_abs_double_vec_inplace(x1_double_a_);
        vec_helper->return_abs_double_vec_inplace(x2_double_a_);
        double ax1aax2a = reduction_double->dot(x1_double_a_, x2_double_a_);
        return( T(ax1aax2a/x1x2) );
    }

    T estimate_condition_blas(T_vec x1_, T_vec x2_)
    {
        
        T x1x2 = std::abs( vec_ops->scalar_prod(x1_, x2_) );
        vec_helper->return_abs_vec(x1_, x1_a_);
        vec_helper->return_abs_vec(x2_, x2_a_);
        T ax1aax2a = vec_ops->scalar_prod(x1_a_, x2_a_);
        return(ax1aax2a/x1x2);
    }

    inline T dot(T_vec d1, T_vec d2)
    {
        return( dot_blas(d1, d2) );
    }
    inline T dot_reduction(T_vec d1, T_vec d2)
    {
        return( vec_ops->scalar_prod(d1, d2) );
    }

    inline T dot_blas(T_vec d1, T_vec d2)
    {
        return( reduction->dot(d1, d2) );
    }

    T dot_exact(T_vec d1, T_vec d2)
    {
        if(use_exact_dot == 1)
        {
            exact_dot->set_arrays(sz, d1, d2);
            return( exact_dot->dot_exact() );
        }
        else if(use_exact_dot == 2)
        {
            vec_ops->get(d1, d1_c);
            vec_ops->get(d2, d2_c);
            exact_dot->set_arrays(sz, d1_c, d2_c);
            return( exact_dot->dot_exact() );            
        }
        else
        {
            return T(0.0);
        }
    }



};

#endif