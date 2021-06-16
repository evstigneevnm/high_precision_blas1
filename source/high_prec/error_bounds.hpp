#ifndef __ERROR_BOUNDS_HPP__
#define __ERROR_BOUNDS_HPP__

#include <limits>
#include <stdio.h>
#include <cmath>




/**
 * @brief      This class generatrs error bounds according to the paper.
 *
 * @tparam     T     { base type of floating points, i.e. float or double }
 */
template<class T>
class error_bounds
{
public:
    error_bounds()
    {
        eps =  macheps();
    }
    ~error_bounds()
    {

    }

    long double macheps()
    {
        return static_cast<long double>( std::numeric_limits<T>::epsilon() );
    }


    void generate(size_t n_, T condition_number_, size_t num_blocks_)
    {
        generate_bounds_real_sum(n_, condition_number_, num_blocks_);
        generate_bounds_real_dot(n_, condition_number_, num_blocks_);

    }

    void generate_real_sum(size_t n_,T condition_number_, size_t num_blocks_)
    {
        update_real_sums(n_, condition_number_, num_blocks_);
    }

    void generate_real_dot(size_t n_, T condition_number_, size_t num_blocks_)
    {
        update_real_dots(n_, condition_number_, num_blocks_);
    }
    void generate_complex_dot(size_t n_, T condition_number_, size_t num_blocks_)
    {
        update_complex_dots(n_, condition_number_, num_blocks_);
    }
        



    struct sum_
    {    
        struct real_
        {
            struct base_
            {
                
                long double sequential;
                long double block_parallel;
                long double pairwise_parallel;
            } base;
            struct compensated_
            {
                long double sequential;
                long double block_parallel;
                long double pairwise_parallel;                
            } compensated;
        } real;
        struct complex_
        {
            struct base_
            {
                long double sequential;
                long double block_parallel;
                long double pairwise_parallel;
            } base;
            struct compensated_
            {
                long double sequential;
                long double block_parallel;
                long double pairwise_parallel;                
            } compensated;
        } complex;    
    } sum;


    struct dot_
    {    
        struct real_
        {
            struct base_
            {
                long double sequential;
                long double block_parallel;
                long double pairwise_parallel;
            } base;
            struct compensated_
            {
                long double sequential;
                long double block_parallel;
                long double pairwise_parallel;                
            } compensated;
        } real;
        struct complex_
        {
            struct base_
            {
                long double sequential;
                long double block_parallel;
                long double pairwise_parallel;
            } base;
            struct compensated_
            {
                long double sequential;
                long double block_parallel;
                long double pairwise_parallel;                
            } compensated;
        } complex;    
    } dot;

private:
    long double eps;


    long double gamma(size_t n)
    {
        long double gamma_ = static_cast<long double>(n)*eps/(1.0-static_cast<long double>(n)*eps);
        return gamma_;
    }


    void simple_sum_bound(size_t n, T condition_number)
    {
        long double gamma_nm1 = gamma(n-1);
        long double res = gamma_nm1*(static_cast<long double>(condition_number));
        sum.real.base.sequential = res;
    }
    void block_sum_bound(size_t b, size_t k, T condition_number)
    {
        long double gamma_km1 = gamma(k-1);
        long double gamma_bm1 = gamma(b-1);
        sum.real.base.block_parallel = (gamma_bm1 + gamma_km1 + gamma_bm1*gamma_km1)*(static_cast<long double>(condition_number));
    }
    void pairwise_sum_bound(size_t n, T condition_number)
    {
        size_t log2n = std::ceil(std::log2((double)n ));
        long double gamma_log2n = gamma(log2n);
        sum.real.base.pairwise_parallel = gamma_log2n*(static_cast<long double>(condition_number));
    }


    void simple_dot_bound_real(size_t n, T condition_number)
    {
        long double gamma_nm1 = gamma(n);
        long double res = 0.5*gamma_nm1*(static_cast<long double>(condition_number));
        dot.real.base.sequential = res;
    }
    void simple_dot_bound_complex(size_t n, T condition_number)
    {
        
        long double res = 0.5*(gamma(n)+gamma(2)+gamma(n)*gamma(2))*(static_cast<long double>(condition_number));
        dot.complex.base.sequential = res;
    }    
    void block_dot_bound_real(size_t b, size_t k, T condition_number)
    {
        long double gamma_km1 = gamma(k);
        long double gamma_bm1 = gamma(b);
        dot.real.base.block_parallel = 0.5*(gamma_bm1 + gamma_km1 + gamma_bm1*gamma_km1)*(static_cast<long double>(condition_number));
    }
    void block_dot_bound_complex(size_t b, size_t k, T condition_number)
    {
        long double gamma_km1 = gamma(k+2);
        long double gamma_bm1 = gamma(b+2);
        dot.complex.base.block_parallel = 0.5*(gamma_bm1 + gamma_km1 + gamma_bm1*gamma_km1)*(static_cast<long double>(condition_number));
    }    
    void pairwise_dot_bound_real(size_t n, T condition_number)
    {
        size_t log2n = std::ceil(std::log2((double)n ))+1;
        long double gamma_log2n = gamma(log2n);
        dot.real.base.pairwise_parallel = 0.5*gamma_log2n*(static_cast<long double>(condition_number));
    }
    void pairwise_dot_bound_complex(size_t n, T condition_number)
    {
        size_t log2n = std::ceil(std::log2((double)n ))+3;
        long double gamma_log2n = gamma(log2n);
        dot.complex.base.pairwise_parallel = 0.5*gamma_log2n*(static_cast<long double>(condition_number));
    }



    void comp_pairwise_sum_bound(size_t n, T condition_number)
    {
        size_t log2n = std::ceil(std::log2(static_cast<double>(n) ))+1;
        long double gamma_log2n = gamma(log2n);
        sum.real.compensated.pairwise_parallel = eps + gamma_log2n*gamma_log2n*(static_cast<long double>(condition_number));
    }

    void ogita_sum_bound(size_t n, T condition_number)
    {
        long double gamma_n = gamma(n-1);
        sum.real.compensated.sequential = eps + gamma_n*gamma_n*(static_cast<long double>(condition_number));
    }

    void comp_block_sum_bound(size_t b, size_t k, T condition_number)
    {
        long double gamma_k = gamma(k-1);
        long double gamma_b = gamma(b-1);
        sum.real.compensated.block_parallel = eps + (2.0*gamma_k*gamma_k+gamma_b*gamma_b+gamma_k*gamma_b+2.0*gamma_k*gamma_b*gamma_b)*(static_cast<long double>(condition_number));
    }

    void comp_pairwise_dot_bound_real(size_t n, T condition_number)
    {
        size_t log2n = std::ceil(std::log2(static_cast<double>(n) ))+2;
        long double gamma_log2n = gamma(log2n);
        dot.real.compensated.pairwise_parallel = eps + 0.5*gamma_log2n*gamma_log2n*(static_cast<long double>(condition_number));
    }
    void comp_pairwise_dot_bound_complex(size_t n, T condition_number)
    {
        size_t log2n = std::ceil(std::log2(static_cast<double>(n) ))+3;
        long double gamma_log2n = gamma(log2n);
        dot.complex.compensated.pairwise_parallel = eps + 0.5*gamma_log2n*gamma_log2n*(static_cast<long double>(condition_number));
    }
    void ogita_dot_bound_real(size_t n, T condition_number)
    {
        long double gamma_n = gamma(n);
        dot.real.compensated.sequential = eps + 0.5*gamma_n*gamma_n*(static_cast<long double>(condition_number));
    }
    void ogita_dot_bound_complex(size_t n, T condition_number)
    {
        long double gamma_n = gamma(n+2);
        dot.complex.compensated.sequential = eps + 0.5*gamma_n*gamma_n*(static_cast<long double>(condition_number));
    }
    void comp_block_dot_bound_real(size_t b, size_t k, T condition_number)
    {
        long double gamma_k = gamma(k);
        long double gamma_b = gamma(b);
        dot.real.compensated.block_parallel = eps + 0.5*(2.0*gamma_k*gamma_k+gamma_b*gamma_b+gamma_k*gamma_b+gamma_k*gamma_b*gamma_b)*(static_cast<long double>(condition_number));
    }

    void comp_block_dot_bound_complex(size_t b, size_t k, T condition_number)
    {
        long double gamma_k = gamma(k+1);
        long double gamma_b = gamma(b+3);
        dot.complex.compensated.block_parallel = eps + 0.5*(2.0*gamma_k*gamma_k+gamma_b*gamma_b+gamma_k*gamma_b+gamma_k*gamma_b*gamma_b)*(static_cast<long double>(condition_number));
    }
    void update_real_sums(size_t n, T condition_number, size_t k)
    {
        size_t b = ceil(1.0*n/(1.0*k) );
        simple_sum_bound(n, condition_number);
        block_sum_bound(b, k, condition_number);
        pairwise_sum_bound(n, condition_number);
        comp_pairwise_sum_bound(n,condition_number);
        ogita_sum_bound(n, condition_number);
        comp_block_sum_bound(b, k,condition_number);
    }
    void update_real_dots(size_t n, T condition_number, size_t k)
    {
        size_t b = ceil(1.0*n/(1.0*k) );
        simple_dot_bound_real(n, condition_number);
        block_dot_bound_real(b, k, condition_number);
        pairwise_dot_bound_real(n, condition_number);
        comp_pairwise_dot_bound_real(n,condition_number);
        ogita_dot_bound_real(n, condition_number);
        comp_block_dot_bound_real(b, k,condition_number);
    }

    void update_complex_dots(size_t n, T condition_number, size_t k)
    {
        size_t b = ceil(1.0*n/(1.0*k) );
        simple_dot_bound_complex(n, condition_number);
        block_dot_bound_complex(b, k, condition_number);
        pairwise_dot_bound_complex(n, condition_number);
        comp_pairwise_dot_bound_complex(n,condition_number);
        ogita_dot_bound_complex(n, condition_number);
        comp_block_dot_bound_complex(b, k,condition_number);
    }

};


#endif