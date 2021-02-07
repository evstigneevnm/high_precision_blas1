//some test for all implemented vector operations;
#include <cmath>
#include <limits>
#include <iostream>
#include <cstdio>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <common/cpu_vector_operations.h>
#include <common/threaded_reduction.h>
#include <high_prec/dot_product_gmp.hpp>
#include <high_prec/sum_gmp.hpp>
#include <common/gpu_reduction.h>
#include <common/testing/gpu_reduction_ogita.h>
#include <generate_vector_pair.hpp>
#include <chrono>
#include <thrust/complex.h>


int main(int argc, char const *argv[])
{
    
    using T = TYPE;
    using TC = thrust::complex<T>;
    using gpu_vector_operations_complex_t = gpu_vector_operations<TC>;
    using cpu_vector_operations_complex_t = cpu_vector_operations<TC>;
    using gpu_vector_operations_real_t = gpu_vector_operations<T>;
    using cpu_vector_operations_real_t = cpu_vector_operations<T>;

    using T_vec = gpu_vector_operations_real_t::vector_type;
    using TC_vec = gpu_vector_operations_complex_t::vector_type;

    using gpu_reduction_real_t = gpu_reduction<T, T_vec>;

    using gpu_reduction_ogita_real_t = gpu_reduction_ogita<T, T_vec>;
    using gpu_reduction_ogita_complex_t = gpu_reduction_ogita<TC, TC_vec>;

    using dot_exact_t = dot_product_gmp<T, T_vec>;
    using sum_exact_t = sum_gmp<T, T_vec>;


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
    gpu_vector_operations_real_t gR_vecs(vec_size, CUBLAS);
    cpu_vector_operations_real_t cR_vecs(vec_size, dot_prod_type);
    gpu_vector_operations_complex_t gC_vecs(vec_size, CUBLAS);
    
    gpu_reduction_real_t reduction_real(vec_size);
    gpu_reduction_ogita_real_t reduciton_ogita_real(vec_size);
    gpu_reduction_ogita_complex_t reduciton_ogita_complex(vec_size);

    

    unsigned int exact_bits = 1024;
    // dot_exact_t dp_ref(exact_bits);
    // sum_exact_t s_ref(exact_bits);

    TC *u1_d;
    TC *u2_d;
    TC *u1_c;
    TC *u2_c;
    


    gC_vecs.init_vector(u1_d); gC_vecs.init_vector(u2_d);
    gC_vecs.start_use_vector(u1_d); gC_vecs.start_use_vector(u2_d);
    


    printf("using vectors of size = %i\n", vec_size);

    // min_max_t min_max = reduction.min_max(u3_d);
    // printf("min = %lf, max = %lf \n", min_max.first, min_max.second);
    // T sum = reduction.sum(u1_d);
    // printf("sum = %lf \n", sum);
    // printf("mean= %lf \n", sum/T(vec_size));

    gC_vecs.assign_scalar(TC( 1.0,-1.0), u1_d);
    gC_vecs.assign_scalar(TC( 5.0, 2.0), u2_d);
    gC_vecs.set_value_at_point(TC(-1.9999999276e5,1.9999999276e5), vec_size-10, u1_d);
    gC_vecs.set_value_at_point(TC(3.987654321e4, 3.987654321e4), vec_size-7, u1_d);
    

    T norm_u1 = gC_vecs.norm(u1_d);
    T norm_u2 = gC_vecs.norm(u2_d);
    gC_vecs.use_high_precision();
    T norm_u1_o = gC_vecs.norm(u1_d);
    T norm_u2_o = gC_vecs.norm(u2_d);
    gC_vecs.use_standard_precision();
    printf("||u1|| = %.24le, ||u2|| = %.24le \n", double(norm_u1), double(norm_u2) );
    printf("||u1||o= %.24le, ||u2||o= %.24le \n", double(norm_u1_o), double(norm_u2_o) );
    printf("  err1 = %.24le,   err2 = %.24le \n", double(std::abs(norm_u1_o - norm_u1)), double(std::abs(norm_u2_o - norm_u2)) );

// save to host 

    if(operation_type == 0 || operation_type == 2)
    {
        printf("========================= dotC =========================\n");    
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start); 
        TC dot_prod_L = gC_vecs.scalar_prod(u1_d, u2_d);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        if(dot_prod_L.imag() < T(0.0) )
            printf("dot_L = %.24le%.24lei time = %f ms\n", double(dot_prod_L.real()), double(dot_prod_L.imag()), milliseconds);
        else
            printf("dot_L = %.24le+%.24lei time = %f ms\n", double(dot_prod_L.real()), double(dot_prod_L.imag()), milliseconds);
        
        gC_vecs.use_high_precision();
        auto start_ch = std::chrono::steady_clock::now();
        // TC dot_prod_ogita_G = reduciton_ogita_complex.dot(u1_d, u2_d);
        TC dot_prod_ogita_G = gC_vecs.scalar_prod(u1_d, u2_d);
        cudaDeviceSynchronize();
        auto finish_ch = std::chrono::steady_clock::now();
        auto elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();        
        if(dot_prod_ogita_G.imag() < T(0.0) )
            printf("dot_OG= %.24le%.24lei time = %f ms\n", double(dot_prod_ogita_G.real()), double(dot_prod_ogita_G.imag()), elapsed_mseconds);
        else
            printf("dot_OG= %.24le+%.24lei time = %f ms\n", double(dot_prod_ogita_G.real()), double(dot_prod_ogita_G.imag()), elapsed_mseconds);
        gC_vecs.use_standard_precision();
        printf("d_dot  = %.24le \n", std::abs<T>(dot_prod_L - dot_prod_ogita_G) );

    }
    if(operation_type == 0 || operation_type == 1)
    {
        printf("========================= sum =========================\n");
        //sum reduction
        auto start_ch = std::chrono::steady_clock::now();
        T asum_L = gC_vecs.absolute_sum(u2_d);
        auto finish_ch = std::chrono::steady_clock::now();
        auto elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
        printf("asum_L = %.24le, time_wall = %lf ms\n", double(asum_L), elapsed_mseconds);

        T asum_ogita_G = 0.0;

        start_ch = std::chrono::steady_clock::now();
        asum_ogita_G = gC_vecs.absolute_sum(u2_d);
        finish_ch = std::chrono::steady_clock::now();
        elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
        printf("asum_OgG= %.24le, time_wall = %lf ms\n", double(asum_ogita_G), elapsed_mseconds);

        printf("d_asum  = %.24le \n", std::abs<T>(asum_ogita_G - asum_L) );

 /*       if(use_ref)
        {

            s_ref.set_array(vec_size, u2_c);
            
            start_ch = std::chrono::steady_clock::now();
            threaded_reduce.use_high_prec();
            T sum_ogita_C_th = threaded_reduce.sum(u2_c);
            finish_ch = std::chrono::steady_clock::now();
            elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
            if(sum_C_th<T(0.0))
                printf("sum_OgC=%.24le, time_wall = %lf ms\n", double(sum_C_th), elapsed_mseconds);
            else
                printf("sum_OgC= %.24le, time_wall = %lf ms\n", double(sum_C_th), elapsed_mseconds);
            
            T ref_exact = s_ref.sum_exact();

            T error_sum_G = s_ref.get_error_relative(sum_G);
            T error_sum_ogita_G = s_ref.get_error_relative(sum_ogita_G);
            T error_sum_C_th = s_ref.get_error_relative(sum_C_th);
            T error_sum_ogita_C_th = s_ref.get_error_relative(sum_ogita_C_th);
            
            

            if(ref_exact<T(0.0))
                printf("ref    =%.24le\n", double(ref_exact)); 
            else
                printf("ref    = %.24le\n", double(ref_exact)); 


            printf("mantisa: \033[0;31mX.123456789123456789\033[0m\n");
            printf("ref    = "); s_ref.print_res();
            printf("err_G  = %.24le, err_Ogita_G = %.24le, err_C_th = %.24le, err_Ogita_C_th = %.24le  \n", double(error_sum_G), double(error_sum_ogita_G), double(error_sum_C_th), double(error_sum_ogita_C_th) );

        }
*/
    }
    
    gC_vecs.free_vector(u1_d); gC_vecs.free_vector(u2_d);

    delete CUBLAS;

    return 0;
}
