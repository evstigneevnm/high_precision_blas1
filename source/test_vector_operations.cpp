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
#include <common/threaded_dot_product.h>
#include <dot_product_gmp.hpp>
#include <sum_gmp.hpp>
#include <common/gpu_reduction.h>
#include <common/testing/gpu_reduction_ogita.h>
#include <generate_vector_pair.hpp>
#include <chrono>

int main(int argc, char const *argv[])
{
    
    using T = TYPE;
    using complex = thrust::complex<T>;
    using gpu_vector_operations_t = gpu_vector_operations<T>;
    using cpu_vector_operations_t = cpu_vector_operations<T>;
    using T_vec = gpu_vector_operations_t::vector_type;
    using gpu_reduction_t = gpu_reduction<T, T_vec>;
    using gpu_reduction_ogita_t = gpu_reduction_ogita<T, T_vec>;    
    using min_max_t = gpu_reduction_t::min_max_t;
    using dot_exact_t = dot_product_gmp<T, T_vec>;
    using sum_exact_t = sum_gmp<T, T_vec>;
    using generate_vector_pair_t = generate_vector_pair<gpu_vector_operations_t, dot_exact_t, gpu_reduction_t>;
    using threaded_dot_t = threaded_dot_prod<T, T_vec>;

    if(argc != 7)
    {
        std::cout << argv[0] << " G o N ref C wh; where: " << std::endl;
        std::cout << "  'G' is the GPU PCI-bus number or -1 for selection;" << std::endl;
        std::cout << "  'o' is the type of operations type method (0-naive, 1-ogita);" << std::endl;
        std::cout << "  'N' is the vector size; " << std::endl;
        std::cout << "  'ref' is the switch to calculate reference solution (ref = 1/0); " << std::endl;
        std::cout << "  'C' is the condition number;" << std::endl;
        std::cout << "  'wh' is the operation to be execution (0 - all (sum&dot), 1 - sum, 2 - dot)." << std::endl;        
        return 0;
    }
    int gpu_pci_id = atoi(argv[1]);
    int dot_prod_type = atoi(argv[2]);
    int vec_size = atoi(argv[3]);
    bool use_ref = atoi(argv[4]);
    T cond_number_ = atof(argv[5]);
    int operation_type = atoi(argv[6]);

    init_cuda(gpu_pci_id);

    cublas_wrap *CUBLAS = new cublas_wrap(true);
    gpu_vector_operations_t g_vecs(vec_size, CUBLAS);
    cpu_vector_operations_t c_vecs(vec_size, dot_prod_type);
    gpu_reduction_t reduction(vec_size);
    gpu_reduction_ogita_t reduciton_ogita(vec_size);
    threaded_dot_t threaded_dot(vec_size, -1, dot_prod_type);


    unsigned int exact_bits = 1024;
    dot_exact_t dp_ref(exact_bits);
    sum_exact_t s_ref(exact_bits);

    T *u1_d;
    T *u2_d;
    T *u3_d;
    T *u1_c;
    T *u2_c;
    


    g_vecs.init_vector(u1_d); g_vecs.init_vector(u2_d); g_vecs.init_vector(u3_d);
    g_vecs.start_use_vector(u1_d); g_vecs.start_use_vector(u2_d); g_vecs.start_use_vector(u3_d);
    c_vecs.init_vector(u1_c); c_vecs.init_vector(u2_c); 
    c_vecs.start_use_vector(u1_c); c_vecs.start_use_vector(u2_c);


    printf("using vectors of size = %i\n", vec_size);
    g_vecs.assign_scalar(T(0.0), u3_d);
    g_vecs.set_value_at_point(T(-120.0), 1234, u3_d);
    g_vecs.set_value_at_point(T(120.0), vec_size/2, u3_d);

    // min_max_t min_max = reduction.min_max(u3_d);
    // printf("min = %lf, max = %lf \n", min_max.first, min_max.second);
    // T sum = reduction.sum(u1_d);
    // printf("sum = %lf \n", sum);
    // printf("mean= %lf \n", sum/T(vec_size));

    generate_vector_pair_t generator(&g_vecs, &dp_ref, &reduction);
    T cond_estimste = generator.generate(u1_d, u2_d, cond_number_);
    printf("condition estimate = %le\n", cond_estimste);
    
    T norm_u1 = g_vecs.norm(u1_d);
    T norm_u2 = g_vecs.norm(u2_d);
    printf("||u1|| = %.24le, ||u2|| = %.24le \n", double(norm_u1), double(norm_u2) );
// save to host 
    g_vecs.get(u1_d, u1_c);
    g_vecs.get(u2_d, u2_c);

    if(operation_type == 0 || operation_type == 2)
    {
        printf("========================= dot =========================\n");    
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start); 
        T dot_prod_1 = g_vecs.scalar_prod(u1_d, u2_d);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("dot_L = %.24le, time = %f ms\n", double(dot_prod_1), milliseconds);
        
        auto start_ch = std::chrono::steady_clock::now();
        cudaEventRecord(start);
        T dot_prod_3 = reduction.dot(u1_d, u2_d);
        cudaDeviceSynchronize();
        auto finish_ch = std::chrono::steady_clock::now();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        // T elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<T> >(finish_ch - start_ch).count();
        T elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();

        printf("dot_G = %.24le, time = %f ms, time_wall = %lf ms\n", double(dot_prod_3), milliseconds, elapsed_mseconds);

        start_ch = std::chrono::steady_clock::now();
        T dot_prod_C_th = threaded_dot.execute(u1_c, u2_c);
        finish_ch = std::chrono::steady_clock::now();
        elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
        printf("dot_Ct= %.24le, time_wall = %lf ms\n", double(dot_prod_C_th), double(elapsed_mseconds) );
        if(use_ref)
        {

            dp_ref.set_arrays(vec_size, u1_c, u2_c);
            
            start_ch = std::chrono::steady_clock::now();
            T dot_prod_2 = c_vecs.scalar_prod(u1_c, u2_c);
            finish_ch = std::chrono::steady_clock::now();
            elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
            printf("dot_C = %.24le, time_wall = %lf ms\n", double(dot_prod_2), double(elapsed_mseconds) );
            
            T ref_exact = dp_ref.dot_exact();
            T error_dot_L = dp_ref.get_error_T(dot_prod_1);
            T error_dot_G = dp_ref.get_error_T(dot_prod_3);
            T error_dot_C = dp_ref.get_error_T(dot_prod_2);
            T error_dot_C_th = dp_ref.get_error_T(dot_prod_C_th);

            printf("ref   = ");
            dp_ref.print_res();       
            printf("ref   = %.24le\n", double(ref_exact)); 
            printf("mantisa:_.123456789123456789\n");
            printf("err_L = %.24le; err_G = %.24le; err_Ct = %.24le; err_C = %.24le\n", double(error_dot_L), double(error_dot_G), double(error_dot_C_th), double(error_dot_C) );
        }
    }
    if(operation_type == 0 || operation_type == 1)
    {
        printf("========================= sum =========================\n");
        //sum reduction
        auto start_ch = std::chrono::steady_clock::now();
        T asum_L = g_vecs.absolute_sum(u1_d);
        auto finish_ch = std::chrono::steady_clock::now();
        auto elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
        printf("asum_L = %.24le, time_wall = %lf ms\n", double(asum_L), elapsed_mseconds);
        start_ch = std::chrono::steady_clock::now();
        T asum_G = reduction.asum(u1_d);
        finish_ch = std::chrono::steady_clock::now();
        elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
        printf("asum_G = %.24le, time_wall = %lf ms\n", double(asum_G), elapsed_mseconds);
        start_ch = std::chrono::steady_clock::now();
        T sum_G = reduction.sum(u1_d);
        finish_ch = std::chrono::steady_clock::now();
        elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
        printf("sum_G  = %.24le, time_wall = %lf ms\n", double(sum_G), elapsed_mseconds);    

        start_ch = std::chrono::steady_clock::now();
        T sum_ogita_G = reduciton_ogita.sum(u1_d);
        finish_ch = std::chrono::steady_clock::now();
        elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
        printf("sum_OgG= %.24le, time_wall = %lf ms\n", double(sum_ogita_G), elapsed_mseconds);
        

        if(use_ref)
        {

            s_ref.set_array(vec_size, u1_c);
            
            // start_ch = std::chrono::steady_clock::now();
            // T dot_prod_2 = c_vecs.scalar_prod(u1_c, u2_c);
            // finish_ch = std::chrono::steady_clock::now();
            // elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
            // printf("dot_C = %.24le, time_wall = %lf ms\n", double(dot_prod_2), double(elapsed_mseconds) );
            
            T ref_exact = s_ref.sum_exact();

            T error_sum_G = s_ref.get_error_T(sum_G);
            T error_sum_ogita_G = s_ref.get_error_T(sum_ogita_G);
            

            printf("ref   = ");
            s_ref.print_res();       
            printf("ref   = %.24le\n", double(ref_exact)); 
            printf("mantisa:X.123456789123456789\n");
            printf("err_G = %.24le, err_Ogita_G = %.24le \n", double(error_sum_G), double(error_sum_ogita_G) );

        }
    }
    
    g_vecs.free_vector(u1_d); g_vecs.free_vector(u2_d); g_vecs.free_vector(u3_d); 
    c_vecs.free_vector(u1_c); c_vecs.free_vector(u2_c);

    delete CUBLAS;

    return 0;
}