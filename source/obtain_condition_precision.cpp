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
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <common/cpu_vector_operations.h>
#include <common/threaded_reduction.h>
#include <common/ogita/gpu_reduction_ogita.h>
#include <high_prec/dot_product_gmp.hpp>
#include <common/gpu_reduction.h>
#include <generate_vector_pair.hpp>
#include <chrono>
#include <string>
#include <fstream>
#include <high_prec/error_bounds.hpp>

template<class T>
T normalize_error(T error_)
{
    return( error_>T(1.0)?1.0:error_ );
}





template<class T>
std::string return_type_name(T some_var);

template<>
std::string return_type_name(float some_var)
{
    return("float");
}
template<>
std::string return_type_name(double some_var)
{
    return("double");
}


int main(int argc, char const *argv[])
{
    
    using T = TYPE;
    using complex = thrust::complex<T>;
    using gpu_vector_operations_t = gpu_vector_operations<T>;
    using cpu_vector_operations_t = cpu_vector_operations<T>;
    using gpu_vector_operations_double_t = gpu_vector_operations<double>;    
    using T_vec = gpu_vector_operations_t::vector_type;
    using gpu_reduction_t = gpu_reduction<T, T_vec>;
    using gpu_reduction_ogita_t = gpu_reduction_ogita<T, T_vec>; 
    using min_max_t = gpu_reduction_t::min_max_t;
    using dot_exact_t = dot_product_gmp<T, T_vec>;
    using dot_exact_double_t = dot_product_gmp<double, double*>;
    using generate_vector_pair_t = generate_vector_pair<gpu_vector_operations_t, gpu_vector_operations_double_t, dot_exact_double_t>;
    using threaded_reduction_t = threaded_reduction<T, T_vec>;
    using error_bounds_t = error_bounds<T>;    


    if(argc != 7)
    {
        std::cout << argv[0] << " G N C dC S host; where: " << std::endl;
        std::cout << "  'G' is the GPU PCI-bus number or -1 for selection; " << std::endl;
        std::cout << "  'N' is the vector size; " << std::endl;
        std::cout << "  'C' is the maximum condition number; " << std::endl;
        std::cout << "  'dC' is the condition number multiplication step size; " << std::endl;
        std::cout << "  'S' is the number of executions on each step; " << std::endl;
        std::cout << "  'host' is the char that sets the usage of the reference on GPU (g), CPU (c) or lower order compensation estimate (o). " << std::endl;
        return 0;
    }
    int gpu_pci_id = atoi(argv[1]);
    int vec_size = atoi(argv[2]);
    T cond_number_max = atof(argv[3]);
    T cond_step_ = atof(argv[4]);
    int executions_step = atof(argv[5]);
    char exact_host =argv[6][0];
    std::string type_name = return_type_name<T>(cond_number_max);
    init_cuda(gpu_pci_id);
    
    int dot_prod_type_initial = 0;
    dot_exact_t dp_ref(1024);
    dot_exact_double_t dp_double(1024);

    cublas_wrap *CUBLAS_ref = new cublas_wrap(true);
    gpu_vector_operations_t g_vecs(vec_size, CUBLAS_ref);
    cpu_vector_operations_t c_vecs(vec_size, dot_prod_type_initial);
    gpu_reduction_t reduction(vec_size);
    gpu_reduction_ogita_t reduction_ogita(vec_size);
    gpu_vector_operations_double_t g_vecs_double(vec_size, CUBLAS_ref);
    threaded_reduction_t threaded_reduce(vec_size, -1, dot_prod_type_initial);
    error_bounds_t err_bnd;

    std::cout << "Machine epsilon is " << err_bnd.macheps() << " " << std::endl;

    T *u1_d; T *u2_d; T *u1_c; T *u2_c;

    g_vecs.init_vector(u1_d); g_vecs.init_vector(u2_d);
    g_vecs.start_use_vector(u1_d); g_vecs.start_use_vector(u2_d);
    c_vecs.init_vector(u1_c); c_vecs.init_vector(u2_c); 
    c_vecs.start_use_vector(u1_c); c_vecs.start_use_vector(u2_c);
    printf("using vectors of size = %le\n", double(vec_size) );
    generate_vector_pair_t generator(&g_vecs, &g_vecs_double, &dp_double);
    
    if(exact_host == 'g')
    {
        generator.dot_exact_cuda();
    }
    else if (exact_host == 'c')
    {
        generator.dot_exact();
    }


    T cond_number = T(1.0);

    std::string f_name = type_name + "_vec_size" + std::to_string(vec_size) + "_Cmax" + std::to_string(std::round(cond_number_max)) + ".dat";
    std::ofstream f(f_name.c_str(), std::ofstream::out);
    if (!f) throw std::runtime_error("error while opening file for writing: " + f_name);

    int count_above = executions_step;

    while( count_above > 0)
    {
        if(cond_number > cond_number_max)
        {
            count_above--;
        }        
        for(int le = 0; le < executions_step; le++)
        {
            T cond_estimste = generator.generate(u1_d, u2_d, cond_number);
            printf("condition estimate = %le\n", cond_estimste);
            g_vecs.use_standard_precision();
            T dot_prod_BLAS = g_vecs.scalar_prod(u1_d, u2_d);
            T dot_prod_reduct = reduction.dot(u1_d, u2_d);
            //T dot_prod_reduct_ogita = reduction_ogita.dot(u1_d, u2_d);
            g_vecs.use_high_precision();
            T dot_prod_reduct_ogita = g_vecs.scalar_prod(u1_d, u2_d);
            g_vecs.use_standard_precision();
            printf("dot_L  = %.24le \n", double(dot_prod_BLAS) );  
            printf("dot_G  = %.24le \n", double(dot_prod_reduct) );        
            
            g_vecs.get(u1_d, u1_c);
            g_vecs.get(u2_d, u2_c);
            
            threaded_reduce.use_standard_precision();            
            T dot_prod_th = threaded_reduce.dot(u1_c, u2_c);
            threaded_reduce.use_high_precision();
            T dot_prod_th_H = threaded_reduce.dot(u1_c, u2_c);
            threaded_reduce.use_standard_precision();

            c_vecs.use_standard_precision();
            T dot_prod = c_vecs.scalar_prod(u1_c, u2_c);
            c_vecs.use_high_precision();
            T dot_prod_H = c_vecs.scalar_prod(u1_c, u2_c);            
            c_vecs.use_standard_precision();

            printf("dot_C  = %.24le \n", double(dot_prod) );
            printf("dot_Ct = %.24le \n", double(dot_prod_th) );
            printf("*dot_CH= %.24le \n", double(dot_prod_H) );
            printf("*dotCtH= %.24le \n", double(dot_prod_th_H) );
            printf("*dot_OG= %.24le \n", double(dot_prod_reduct_ogita) );            

            dp_ref.set_arrays(vec_size, u1_c, u2_c);
            T ref_exact = dp_ref.dot_exact();            

            err_bnd.generate_real_dot(vec_size, cond_estimste, 24);

            T error_exact_L = dp_ref.get_error_relative(dot_prod_BLAS);
            T error_exact_G = dp_ref.get_error_relative(dot_prod_reduct);
            T error_exact_ogita_G = dp_ref.get_error_relative(dot_prod_reduct_ogita);
            T error_exact_C = dp_ref.get_error_relative(dot_prod);
            T error_exact_C_H = dp_ref.get_error_relative(dot_prod_H);    
            T error_exact_C_th = dp_ref.get_error_relative(dot_prod_th);
            T error_exact_C_th_H = dp_ref.get_error_relative(dot_prod_th_H);  


            long double simple_bound_ = normalize_error<long double>(err_bnd.dot.real.base.sequential);
            long double pairwise_bound_ = normalize_error<long double>(err_bnd.dot.real.base.pairwise_parallel );
            long double parallel24_bound_ = normalize_error<long double>(err_bnd.dot.real.base.block_parallel);
            
            long double ogita_bound_ = normalize_error<long double>(err_bnd.dot.real.compensated.sequential );
            long double pairwise_comp_bound_ = normalize_error<long double>(err_bnd.dot.real.compensated.pairwise_parallel );
            long double parallel_comp_24_bound_ = normalize_error<long double>(err_bnd.dot.real.compensated.block_parallel);

            // std::cout << "simple:" << simple_bound_ << " pairwise:" << pairwise_bound_ << " parallel24:" << parallel24_bound_ << " ogita:" << ogita_bound_ << " c_pairwise:" << pairwise_comp_bound_ << " c_parallel24:" << parallel_comp_24_bound_ << std::endl;
            

            printf("ref    = %.24le \n", double(ref_exact));        
            printf("mantisa:\033[0;31mX.123456789123456789\033[0m \n");
            printf("err_L  = %.24le | %.24le \nerr_G  = %.24le | %.24le \nerr_Ct = %.24le | %.24le\nerr_C  = %.24le | %.24le \n*err_CH= %.24le | %.24le \n*errCtH= %.24le | %.24le \n*err_GH= %.24le | %.24le\n", double(error_exact_L), double(pairwise_bound_), double(error_exact_G), double(pairwise_bound_), double(error_exact_C_th), double(parallel24_bound_), double(error_exact_C), double(simple_bound_), double(error_exact_C_H), double(ogita_bound_), double(error_exact_C_th_H), double(parallel_comp_24_bound_), double(error_exact_ogita_G),  double(pairwise_comp_bound_));

            if ( !(f << cond_estimste  << " " << normalize_error(error_exact_L) << " " << normalize_error(error_exact_G) << " " << normalize_error(error_exact_ogita_G) << " " << normalize_error(error_exact_C) << " " <<  normalize_error(error_exact_C_H) << " " << normalize_error(error_exact_C_th) << " " << normalize_error(error_exact_C_th_H) << " " << ogita_bound_ << " " << parallel24_bound_ << " " << parallel_comp_24_bound_ << " " << pairwise_bound_ << " " << pairwise_comp_bound_ << " "<< simple_bound_ << std::endl ) )
            {
                throw std::runtime_error("error while writing to file: " + f_name);
            }
            std::cout << std::endl;

        }

        cond_number *= cond_step_;
    }
    f.close();

    c_vecs.free_vector(u1_c); c_vecs.free_vector(u2_c);    
    g_vecs.free_vector(u1_d); g_vecs.free_vector(u2_d);
    

    delete CUBLAS_ref;

    return 0;
}