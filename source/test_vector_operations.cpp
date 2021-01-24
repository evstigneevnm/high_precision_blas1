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
#include <common/gpu_reduction.h>
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
    using min_max_t = gpu_reduction_t::min_max_t;
    using dot_exact_t = dot_product_gmp<T, T_vec>;
    using generate_vector_pair_t = generate_vector_pair<gpu_vector_operations_t, dot_exact_t, gpu_reduction_t>;
    using threaded_dot_t = threaded_dot_prod<T, T_vec>;

    if(argc != 6)
    {
        std::cout << argv[0] << " G o N ref C; where: " << std::endl;
        std::cout << "  'G' is the GPU PCI-bus number or -1 for selection;" << std::endl;
        std::cout << "  'o' is the type of dot product method (0-naive, 1-ogita);" << std::endl;
        std::cout << "  'N' is the vector size; " << std::endl;
        std::cout << "  'ref' is the switch to calculate reference solution (ref = 1/0); " << std::endl;
        std::cout << "  'C' is the condition number." << std::endl;
        return 0;
    }
    int gpu_pci_id = atoi(argv[1]);
    int dot_prod_type = atoi(argv[2]);
    int vec_size = atoi(argv[3]);
    bool use_ref = atoi(argv[4]);
    T cond_number_ = atof(argv[5]);

    init_cuda(gpu_pci_id);

    cublas_wrap *CUBLAS = new cublas_wrap(true);
    gpu_vector_operations_t g_vecs(vec_size, CUBLAS);
    cpu_vector_operations_t c_vecs(vec_size, dot_prod_type);
    gpu_reduction_t reduction(vec_size);
    threaded_dot_t threaded_dot(vec_size, -1, dot_prod_type);

    T *u1_d;
    T *u2_d;
    T *u3_d;
    T *u1_c;
    T *u2_c;
    
    dot_exact_t dp_ref(4096);

    g_vecs.init_vector(u1_d); g_vecs.init_vector(u2_d); g_vecs.init_vector(u3_d);
    g_vecs.start_use_vector(u1_d); g_vecs.start_use_vector(u2_d); g_vecs.start_use_vector(u3_d);
    c_vecs.init_vector(u1_c); c_vecs.init_vector(u2_c); 
    c_vecs.start_use_vector(u1_c); c_vecs.start_use_vector(u2_c);


    printf("using vectors of size = %i\n", vec_size);
    //g_vecs.assign_scalar(T(2.0e-5), u1_d);
    //g_vecs.assign_scalar(T(2.0e-5), u2_d);
    g_vecs.assign_random(u1_d, T(-100), T(100));
    g_vecs.assign_random(u2_d, T(-100), T(100));
    // g_vecs.assign_random(u3_d);

    g_vecs.assign_scalar(T(0.0),u3_d);
    g_vecs.set_value_at_point(T(-120.0), 1234, u3_d);
    g_vecs.set_value_at_point(T(120.0), vec_size/2, u3_d);

    min_max_t min_max = reduction.min_max(u1_d);
    printf("min = %lf, max = %lf \n", min_max.first, min_max.second);
    T sum = reduction.sum(u1_d);
    printf("sum = %lf \n", sum);
    printf("mean= %lf \n", sum/T(vec_size));

    generate_vector_pair_t generator(&g_vecs, &dp_ref, &reduction);
    T cond_estimste = generator.generate(u1_d, u2_d, cond_number_);
    printf("condition estimate = %le\n", cond_estimste);
    
    T norm_u1 = g_vecs.norm(u1_d);
    printf("||u1||= %.24le\n", double(norm_u1));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); 
    T dot_prod_1 = g_vecs.scalar_prod(u1_d, u2_d);
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

    g_vecs.get(u1_d, u1_c);
    g_vecs.get(u2_d, u2_c);

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
        T error_exact_L = dp_ref.get_error(dot_prod_1);
        T error_exact_G = dp_ref.get_error(dot_prod_3);
        T error_exact_C = dp_ref.get_error(dot_prod_2);
        T error_exact_C_th = dp_ref.get_error(dot_prod_C_th);

        printf("ref   = %.24le\n", double(ref_exact));        
        printf("mantisa:_.123456789123456789\n");
        printf("err_L = %.24le; err_G = %.24le; err_Ct = %.24le; err_C = %.24le\n", double(error_exact_L), double(error_exact_G), double(error_exact_C_th), double(error_exact_C) );
    }

    g_vecs.free_vector(u1_d); g_vecs.free_vector(u2_d); g_vecs.free_vector(u3_d); 
    c_vecs.free_vector(u1_c); c_vecs.free_vector(u2_c);

    delete CUBLAS;

    return 0;
}