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
#include <cmath>
#include <vector>
#include <limits>
#include <iostream>
#include <cstdio>
#include <high_prec/sum_gmp.hpp>
#include <random_adapter.hpp>






template<class T, class VecT>
T sum(const VecT& vec)
{

    T res = T(0.0);
    for(auto &x: vec)
    {
        res += x;
    }

    return res;
}

template<class T, class VecT>
T sum2(const VecT& vec)
{

    size_t vec_size = vec.size();
    size_t vec_size_div_2 = vec_size/2;
    T res = T(0.0);
    for(int j=0;j<vec_size_div_2;j++)
    {
        T res_l = T(0.0);
        for(int k=0;k<2;k++)
        {
            res_l += vec.at(2*j+k);
        }        
        res += res_l;
    }

    return res;
}




int main(int argc, char const *argv[])
{
    using T = float;
    using T_vec = std::vector<T>;
    using sum_exact_t = sum_gmp<T, T_vec>;
    using random_adapter_t = random_adapter<T>;
    if(argc != 3)
    {
        std::cout << "testing error bounds for sum-2 method"  << std::endl;
        std::cout << argv[0] << " N m" << std::endl;
        std::cout << "  'N' is the vector size, " << std::endl;
        std::cout << "  'm' is the number of tests. " << std::endl;
        return 0;
    }
    size_t N = std::atoi(argv[1]);
    N = N%2>0?N+1:N;

    size_t attempts = std::atoi(argv[2]);
    unsigned int exact_bits = 1024;
    sum_exact_t s_ref(exact_bits);
    random_adapter_t rand;
    rand.set_uniform_distribution(-150.0, 150.0);

    T_vec vec;
    vec.reserve(N);

    std::vector< std::pair<T,T> > errors;
    std::vector< std::vector<T> > results;

    for(int j=0;j<N;j++)
    {
        vec.push_back( rand.get_uniform() );
    }

    for(int m=0;m<attempts;m++)
    {
        for(int j=0;j<N;j++)
        {
            vec.at(j) = rand.get_uniform() ;
        }
        s_ref.set_array(N, vec);

        T res_1 = sum<T, T_vec>(vec);
        T res_2 = sum2<T, T_vec>(vec);
        T ref_exact = s_ref.sum_exact();
        T err_1 = s_ref.get_error_relative_T(res_1);
        T err_2 = s_ref.get_error_relative_T(res_2);
        T ref_exact_asum = s_ref.asum_exact();
        std::pair<T,T> err = {err_1,err_2};
        errors.push_back(err);
        std::vector<T> res = {ref_exact, res_1, res_2, ref_exact_asum};
        results.push_back(res);
        
    }
    
    
    //std::cout << "machine epsilon = " << FLT_EPSILON << std::endl;
    double gamma_1 = FLT_EPSILON*N/(1.0-FLT_EPSILON*N);
    double gamma_2 = (FLT_EPSILON*(N/2)/(1.0-FLT_EPSILON*(N/2)) +  FLT_EPSILON*2/(1.0-FLT_EPSILON*2) + FLT_EPSILON*(N/2)/(1.0-FLT_EPSILON*(N/2))*FLT_EPSILON*2/(1.0-FLT_EPSILON*2));
    //std::cout << "gamma 1 = " << gamma_1 << " gamma 2 = " << gamma_2 << std::endl;

    T err_1_av = T(0.0);
    T err_2_av = T(0.0);    
    for(int m=0;m<attempts;m++)
    {
    
        if(attempts<10)
        {
            std::cout.precision(12);
            std::cout << std::scientific << results.at(m).at(0) << " " << results.at(m).at(1) << " " << results.at(m).at(2) << " ";
            std::cout << std::scientific << errors.at(m).first << " " << errors.at(m).second << std::endl;
            std::cout << results.at(m).at(3)*gamma_1/std::abs(results.at(m).at(0)) << " " << results.at(m).at(3)*gamma_2/std::abs(results.at(m).at(0)) << std::endl;
        }

        err_1_av += errors.at(m).first;
        err_2_av += errors.at(m).second;
    }
    
    std::cout << std::endl << "machine epsilon = " << FLT_EPSILON << std::endl;
    std::cout << "gamma 1 = " << gamma_1 << " gamma 2 = " << gamma_2 << " gamma_1/gamma_2 = " << gamma_1/gamma_2 << std::endl;

    std::cout << "====================mean errors====================" << std::endl;
    std::cout << std::scientific << err_1_av/attempts << " " << err_2_av/attempts << " " << err_1_av/err_2_av << std::endl;


    return 0;
}