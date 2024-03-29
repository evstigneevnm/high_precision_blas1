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

//some test for all implemented vector operations;
#include <cmath>
#include <limits>
#include <iostream>
#include <cstdio>
#include <vector>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <common/cpu_vector_operations.h>
#include <common/threaded_reduction.h>
#include <high_prec/dot_product_gmp_complex.hpp>
#include <high_prec/dot_product_gmp.hpp>
#include <high_prec/asum_gmp_complex.hpp>
#include <common/gpu_reduction.h>
#include <common/ogita/gpu_reduction_ogita.h>
#include <generate_vector_pair_complex.hpp>
#include <chrono>
#include <thrust/complex.h>

template<class T>
std::vector<T> without_hi_lo(std::vector<T> orig)
{
     std::sort(orig.begin(), orig.end());
     return std::vector<T>(&orig[1], &orig[orig.size()-1]);
}

int main(int argc, char const *argv[])
{
    
    using T = TYPE;
    using TC = thrust::complex<T>;
    using gpu_vector_operations_complex_t = gpu_vector_operations<TC>;
    using cpu_vector_operations_complex_t = cpu_vector_operations<TC>;
    using gpu_vector_operations_real_t = gpu_vector_operations<T>;
    using cpu_vector_operations_real_t = cpu_vector_operations<T>;
    using gpu_vector_operations_double_t = gpu_vector_operations<double>;

    using T_vec = gpu_vector_operations_real_t::vector_type;
    using TC_vec = gpu_vector_operations_complex_t::vector_type;

    using gpu_reduction_real_t = gpu_reduction<T, T_vec>;

    using gpu_reduction_ogita_real_t = gpu_reduction_ogita<T, T_vec>;
    using gpu_reduction_ogita_complex_t = gpu_reduction_ogita<TC, TC_vec>;
    using dot_exact_double_t = dot_product_gmp<double, double*>;

    using dot_exact_t = dot_product_gmp_complex<T, TC_vec>;
    using asum_exact_t = asum_gmp<T, TC_vec>;
    using generate_vector_pair_t = generate_vector_pair_complex<gpu_vector_operations_complex_t,gpu_vector_operations_real_t, gpu_vector_operations_double_t, dot_exact_double_t>;




    if(argc != 8)
    {
        std::cout << argv[0] << " G o N ref C wh rep; where: " << std::endl;
        std::cout << "  'G' is the GPU PCI-bus number or -1 for selection;" << std::endl;
        std::cout << "  'o' is the type of operations type method (0-naive, 1-ogita);" << std::endl;
        std::cout << "  'N' is the vector size; " << std::endl;
        std::cout << "  'ref' is the switch to calculate reference solution (ref = 1/0); " << std::endl;
        std::cout << "  'C' is the condition number;" << std::endl;
        std::cout << "  'wh' is the operation to be execution (0 - all (sum&dot), 1 - sum, 2 - dot);" << std::endl;
        std::cout << "  'rep' is the number of repititions during the tests." << std::endl;        
        return 0;
    }
    int gpu_pci_id = atoi(argv[1]);
    int dot_prod_type = atoi(argv[2]);
    int vec_size = atoi(argv[3]);
    bool use_ref = atoi(argv[4]);
    T cond_number_ = atof(argv[5]);
    int operation_type = atoi(argv[6]);
    int rep = atoi(argv[7]);
    rep = rep + 2;

    init_cuda(gpu_pci_id);

    cublas_wrap *CUBLAS = new cublas_wrap(true);
    gpu_vector_operations_real_t gR_vecs(vec_size, CUBLAS);
    cpu_vector_operations_real_t cR_vecs(vec_size, dot_prod_type);
    gpu_vector_operations_complex_t gC_vecs(vec_size, CUBLAS);
    gpu_vector_operations_double_t g_vecs_double(vec_size, CUBLAS);
    
    gpu_reduction_real_t reduction_real(vec_size);
    gpu_reduction_ogita_real_t reduciton_ogita_real(vec_size);
    gpu_reduction_ogita_complex_t reduciton_ogita_complex(vec_size);

    

    unsigned int exact_bits = 1024;
    dot_exact_t dp_ref(exact_bits);
    asum_exact_t as_ref(exact_bits);
    dot_exact_double_t dp_ref_double(exact_bits);

    TC *u1_d;
    TC *u2_d;
    TC *u1_c;
    TC *u2_c;


    gC_vecs.init_vector(u1_d); gC_vecs.init_vector(u2_d);
    gC_vecs.start_use_vector(u1_d); gC_vecs.start_use_vector(u2_d);
    
    printf("using vectors of size = %i\n", vec_size);
    generate_vector_pair_t generator(&gC_vecs, &gR_vecs, &g_vecs_double, &dp_ref_double);
    auto cond_estimste = generator.generate(u1_d, u2_d, cond_number_);
    std::cout << "condition estimate = " << cond_estimste.first << " " << cond_estimste.second << std::endl;


    T norm_u1 = gC_vecs.norm(u1_d);
    T norm_u2 = gC_vecs.norm(u2_d);
    gC_vecs.use_high_precision();
    T norm_u1_o = gC_vecs.norm(u1_d);
    T norm_u2_o = gC_vecs.norm(u2_d);
    gC_vecs.use_standard_precision();
    printf("||u1|| = %.16le, ||u2|| = %.16le \n", double(norm_u1), double(norm_u2) );
    printf("||u1||o= %.16le, ||u2||o= %.16le \n", double(norm_u1_o), double(norm_u2_o) );
    printf("  err1 = %.16le,   err2 = %.16le \n", double(std::abs(norm_u1_o - norm_u1)), double(std::abs(norm_u2_o - norm_u2)) );

// save to host 

    if(operation_type == 0 || operation_type == 2)
    {
        gC_vecs.use_standard_precision();
        printf("========================= dotC =========================\n");    
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        std::vector<float> wall_times;
        wall_times.reserve(rep);
        
        TC dot_prod_L = 0.0;
        for(int r = 0; r<rep;r++)
        {
            cudaEventRecord(start); 
            dot_prod_L = gC_vecs.scalar_prod(u1_d, u2_d);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            wall_times.push_back(milliseconds);
        }
        wall_times = without_hi_lo<float>(wall_times);
        auto min_max = std::minmax_element(begin(wall_times), end(wall_times));
        auto average = accumulate( wall_times.begin(), wall_times.end(), 0.0)/wall_times.size();
        wall_times.clear();

        if(dot_prod_L.imag() < T(0.0) )
            printf("dot_L = %.16le%.16lei {min, max, ave}time = %f, %f, %f ms\n",  double(dot_prod_L.real()), double(dot_prod_L.imag()), *min_max.first, *min_max.second, average);
        else
            printf("dot_L = %.16le+%.16lei {min, max, ave}time = %f, %f, %f ms\n", double(dot_prod_L.real()), double(dot_prod_L.imag()), *min_max.first, *min_max.second, average);
        
        TC dot_prod_ogita_G = 0.0;
        gC_vecs.use_high_precision();
        for(int r = 0; r<rep;r++)
        {
            auto start_ch = std::chrono::steady_clock::now();
            dot_prod_ogita_G = gC_vecs.scalar_prod(u1_d, u2_d);
            cudaDeviceSynchronize();
            auto finish_ch = std::chrono::steady_clock::now();
            auto elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();      
            wall_times.push_back(elapsed_mseconds);
        }
        wall_times = without_hi_lo<float>(wall_times);        
        min_max = std::minmax_element(begin(wall_times), end(wall_times));
        average = accumulate( wall_times.begin(), wall_times.end(), 0.0)/wall_times.size();
        wall_times.clear();        

        if(dot_prod_ogita_G.imag() < T(0.0) )
            printf("dot_OG= %.16le%.16lei {min, max, ave}time = %f, %f, %f ms\n", double(dot_prod_ogita_G.real()), double(dot_prod_ogita_G.imag()), *min_max.first, *min_max.second, average);
        else
            printf("dot_OG= %.16le+%.16lei {min, max, ave}time = %f, %f, %f ms\n", double(dot_prod_ogita_G.real()), double(dot_prod_ogita_G.imag()), *min_max.first, *min_max.second, average);
        gC_vecs.use_standard_precision();
        printf("d_dot = %.24le \n", std::abs<T>(dot_prod_L - dot_prod_ogita_G) );
 

        if(use_ref)
        {
            u1_c = (TC_vec)malloc(vec_size*sizeof(TC));
            u2_c = (TC_vec)malloc(vec_size*sizeof(TC));

            gC_vecs.get(u1_d, u1_c);
            gC_vecs.get(u2_d, u2_c);

            dp_ref.set_arrays(vec_size, u1_c, u2_c);

            TC ref_exact = dp_ref.dot_exact();
            
            

            printf("ref    =(%.24le,%.24le)\n", double(ref_exact.real()),double(ref_exact.imag()) ); 
         
            TC rel_err_ogita = dp_ref.get_error_relative(dot_prod_ogita_G);
            TC rel_err_L = dp_ref.get_error_relative(dot_prod_L);
            printf("mantisa: \033[0;31mX.123456789123456789\033[0m\n");
            // printf("ref    = "); dp_ref.print_res();    
            std::cout << "error_L = " << rel_err_L << std::endl;
            std::cout << "error_ogita = " << rel_err_ogita  << std::endl;

            free(u1_c);
            free(u2_c);

        }
    }
    if(operation_type == 0 || operation_type == 1)
    {
        printf("========================= sumC =========================\n");
        //sum reduction
        gC_vecs.use_standard_precision();
        
        std::vector<float> wall_times;
        wall_times.reserve(rep);
        T asum_L = 0;
        for(int r = 0; r<rep;r++)
        {
            auto start_ch = std::chrono::steady_clock::now();
            asum_L = gC_vecs.absolute_sum(u2_d);
            cudaDeviceSynchronize();
            auto finish_ch = std::chrono::steady_clock::now();
            auto elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
            wall_times.push_back(elapsed_mseconds);
        }
        wall_times = without_hi_lo<float>(wall_times);
        auto min_max = std::minmax_element(begin(wall_times), end(wall_times));
        auto average = accumulate( wall_times.begin(), wall_times.end(), 0.0)/wall_times.size();
        wall_times.clear(); 

        printf("asum_L = %.16le, {min, max, ave}time = %f, %f, %f ms\n", double(asum_L), *min_max.first, *min_max.second, average);

        T asum_ogita_G = 0.0;
        gC_vecs.use_high_precision();
        for(int r = 0; r<rep;r++)
        {
            auto start_ch = std::chrono::steady_clock::now();
            asum_ogita_G = gC_vecs.absolute_sum(u2_d);
            cudaDeviceSynchronize();
            auto finish_ch = std::chrono::steady_clock::now();
            auto elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
            wall_times.push_back(elapsed_mseconds);
        }
        wall_times = without_hi_lo<float>(wall_times);
        min_max = std::minmax_element(begin(wall_times), end(wall_times));
        average = accumulate( wall_times.begin(), wall_times.end(), 0.0)/wall_times.size();
        wall_times.clear(); 

        printf("asum_OgG=%.16le, {min, max, ave}time = %f, %f, %f ms\n", double(asum_ogita_G), *min_max.first, *min_max.second, average);


        printf("d_asum = %.16le \n", std::abs<T>(asum_ogita_G - asum_L) );
        if(use_ref)
        {
            u1_c = (TC_vec)malloc(vec_size*sizeof(TC));
            u2_c = (TC_vec)malloc(vec_size*sizeof(TC));

            gC_vecs.get(u1_d, u1_c);
            gC_vecs.get(u2_d, u2_c);

            as_ref.set_array(vec_size, u2_c);

            T ref_exact = as_ref.asum_exact();
            
            

            printf("ref    = %.24le\n", double(ref_exact) ); 
         
            T rel_err_ogita = as_ref.get_error_relative(asum_ogita_G);
            T rel_err_L = as_ref.get_error_relative(asum_L);
            printf("mantisa: \033[0;31mX.123456789123456789\033[0m\n");
            // printf("ref    = "); dp_ref.print_res();    
            printf("error_L = %.24le \n", rel_err_L );
            printf("error_ogita = %.24le \n", rel_err_ogita );

            free(u1_c);
            free(u2_c);

        }


    }



    gC_vecs.free_vector(u1_d); gC_vecs.free_vector(u2_d);

    delete CUBLAS;

    return 0;
}
