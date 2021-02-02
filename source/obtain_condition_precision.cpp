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
#include <dot_product_gmp.hpp>
#include <common/gpu_reduction.h>
#include <generate_vector_pair.hpp>
#include <chrono>
#include <string>
#include <fstream>

int main(int argc, char const *argv[])
{
    
    using T = TYPE;
    using complex = thrust::complex<T>;
    using gpu_vector_operations_t = gpu_vector_operations<T>;
    using cpu_vector_operations_t = cpu_vector_operations<T>;
    using T_vec = gpu_vector_operations_t::vector_type;
    using gpu_reduction_t = gpu_reduction<T, T_vec>;
    using min_max_t = gpu_reduction_t::min_max_t;
    using dot_exact_t = dot_product_gmp<T, T_vec>;
    using generate_vector_pair_t = generate_vector_pair<gpu_vector_operations_t, dot_exact_t, gpu_reduction_t>;
    using threaded_reduction_t = threaded_reduction<T, T_vec>;

    if(argc != 6)
    {
        std::cout << argv[0] << " G N C dC S; where: " << std::endl;
        std::cout << "  'G' is the GPU PCI-bus number or -1 for selection; " << std::endl;
        std::cout << "  'N' is the vector size; " << std::endl;
        std::cout << "  'C' is the maximum condition number; " << std::endl;
        std::cout << "  'dC' is the condition number multiplication step size; " << std::endl;
        std::cout << "  'S' is the number of executions on each step. " << std::endl;
        return 0;
    }
    int gpu_pci_id = atoi(argv[1]);
    int vec_size = atoi(argv[2]);
    T cond_number_max = atof(argv[3]);
    T cond_step_ = atof(argv[4]);
    int executions_step = atof(argv[5]);

    init_cuda(gpu_pci_id);
    int dot_prod_type_initial = 0;
    dot_exact_t dp_ref(1024);

    cublas_wrap *CUBLAS_ref = new cublas_wrap(true);
    gpu_vector_operations_t g_vecs(vec_size, CUBLAS_ref);
    cpu_vector_operations_t c_vecs(vec_size, dot_prod_type_initial);
    gpu_reduction_t reduction(vec_size);
    threaded_reduction_t threaded_reduce(vec_size, -1, dot_prod_type_initial);

    T *u1_d; T *u2_d; T *u1_c; T *u2_c;

    g_vecs.init_vector(u1_d); g_vecs.init_vector(u2_d);
    g_vecs.start_use_vector(u1_d); g_vecs.start_use_vector(u2_d);
    c_vecs.init_vector(u1_c); c_vecs.init_vector(u2_c); 
    c_vecs.start_use_vector(u1_c); c_vecs.start_use_vector(u2_c);
    printf("using vectors of size = %le\n", double(vec_size) );
    generate_vector_pair_t generator(&g_vecs, &dp_ref, &reduction);
    

    T cond_number = T(1.0);

    std::string f_name = "vec_size" + std::to_string(vec_size) + "_Cmax" + std::to_string(cond_number_max) + ".dat";
    std::ofstream f(f_name.c_str(), std::ofstream::out);
    if (!f) throw std::runtime_error("error while opening file for writing: " + f_name);

    while( cond_number <= cond_number_max)
    {
        
        for(int le = 0; le < executions_step; le++)
        {
            T cond_estimste = generator.generate(u1_d, u2_d, cond_number);
            printf("condition estimate = %le\n", cond_estimste);
            T dot_prod_BLAS = g_vecs.scalar_prod(u1_d, u2_d);
            T dot_prod_reduct = reduction.dot(u1_d, u2_d);

            printf("dot_L = %.24le \n", double(dot_prod_BLAS) );  
            printf("dot_G = %.24le \n", double(dot_prod_reduct) );          
            
            g_vecs.get(u1_d, u1_c);
            g_vecs.get(u2_d, u2_c);
            
            threaded_reduce.use_normal_prec();            
            T dot_prod_th = threaded_reduce.dot(u1_c, u2_c);
            threaded_reduce.use_high_prec();
            T dot_prod_th_H = threaded_reduce.dot(u1_c, u2_c);

            T dot_prod = c_vecs.scalar_prod(u1_c, u2_c, 0);
            T dot_prod_H = c_vecs.scalar_prod(u1_c, u2_c, 1);            

            printf("dot_C = %.24le \n", double(dot_prod) );
            printf("dot_Ct= %.24le \n", double(dot_prod_th) );
            printf("dot_CH= %.24le \n", double(dot_prod_H) );
            printf("dotCtH= %.24le \n", double(dot_prod_th_H) );

            dp_ref.set_arrays(vec_size, u1_c, u2_c);
            T ref_exact = dp_ref.dot_exact();            

            T error_exact_L = dp_ref.get_error_relative(dot_prod_BLAS);
            T error_exact_G = dp_ref.get_error_relative(dot_prod_reduct);
            T error_exact_C = dp_ref.get_error_relative(dot_prod);
            T error_exact_C_H = dp_ref.get_error_relative(dot_prod_H);    
            T error_exact_C_th = dp_ref.get_error_relative(dot_prod_th);
            T error_exact_C_th_H = dp_ref.get_error_relative(dot_prod_th_H);  

            printf("ref   = %.24le \n", double(ref_exact));        
            printf("mantisa:_.123456789123456789 \n");
            printf("err_L = %.24le \nerr_G = %.24le \nerr_Ct = %.24le \nerr_C = %.24le \nerrCtH = %.24le \nerr_CH = %.24le \n", double(error_exact_L), double(error_exact_G), double(error_exact_C_th), double(error_exact_C), double(error_exact_C_th_H), double(error_exact_C_H) );

            if ( !(f << cond_estimste << " " << error_exact_L << " " << error_exact_G << " " << error_exact_C << " " <<  error_exact_C_H << " " << error_exact_C_th << " " << error_exact_C_th_H << std::endl ) )
            {
                throw std::runtime_error("error while writing to file: " + f_name);
            }

        }

        cond_number *= cond_step_;
    }
    f.close();

    c_vecs.free_vector(u1_c); c_vecs.free_vector(u2_c);    
    g_vecs.free_vector(u1_d); g_vecs.free_vector(u2_d);
    

    delete CUBLAS_ref;

    return 0;
}