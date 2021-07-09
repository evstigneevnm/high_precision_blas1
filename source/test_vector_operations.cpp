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
#include <high_prec/dot_product_cump.hpp>
#include <high_prec/sum_gmp.hpp>
#include <common/gpu_reduction.h>
#include <common/ogita/gpu_reduction_ogita.h>
#include <generate_vector_pair.hpp>
#include <chrono>

template<class T>
std::vector<T> without_hi_lo(std::vector<T> orig)
{
     std::sort(orig.begin(), orig.end());
     return std::vector<T>(&orig[1], &orig[orig.size()-1]);
}


int main(int argc, char const *argv[])
{
    
    using T = TYPE;
    using gpu_vector_operations_t = gpu_vector_operations<T>;
    using cpu_vector_operations_t = cpu_vector_operations<T>;
    using gpu_vector_operations_double_t = gpu_vector_operations<double>;     
    using T_vec = gpu_vector_operations_t::vector_type;
    using gpu_reduction_t = gpu_reduction<T, T_vec>;
    using gpu_reduction_ogita_t = gpu_reduction_ogita<T, T_vec>;    
    using min_max_t = gpu_reduction_t::min_max_t;
    using dot_exact_t = dot_product_gmp<T, T_vec>;
    using sum_exact_t = sum_gmp<T, T_vec>;
    using dot_exact_double_t = dot_product_gmp<double, double*>;
    using generate_vector_pair_t = generate_vector_pair<gpu_vector_operations_t, gpu_vector_operations_double_t, dot_exact_double_t>;    
    using threaded_reduction_t = threaded_reduction<T, T_vec>;
    using dot_product_cump_t = dot_product_cump<T, T_vec>;

    if(argc != 9)
    {
        std::cout << argv[0] << " G o N ref C wh p rep; where: " << std::endl;
        std::cout << "  'G' is the GPU PCI-bus number or -1 for selection;" << std::endl;
        std::cout << "  'o' is the type of operations type method (0-naive, 1-ogita);" << std::endl;
        std::cout << "  'N' is the vector size; " << std::endl;
        std::cout << "  'ref' is the switch to calculate reference solution (ref = 1/0); " << std::endl;
        std::cout << "  'C' is the condition number;" << std::endl;
        std::cout << "  'wh' is the operation to be execution (0 - all (sum&dot), 1 - sum, 2 - dot);" << std::endl;        
        std::cout << "  'p' is the high precision dot product (g - gpu, c - cpu);" << std::endl;
        std::cout << "  'rep' is the number of repititions during the tests." << std::endl;        
        return 0;
    }
    int gpu_pci_id = atoi(argv[1]);
    int dot_prod_type = atoi(argv[2]);
    int vec_size = atoi(argv[3]);
    bool use_ref = atoi(argv[4]);
    T cond_number_ = atof(argv[5]);
    int operation_type = atoi(argv[6]);
    char host_or_device = argv[7][0];
    int rep = atoi(argv[8]);
    rep = rep + 2;

    init_cuda(gpu_pci_id);

    cublas_wrap *CUBLAS = new cublas_wrap(true);
    gpu_vector_operations_t g_vecs(vec_size, CUBLAS);
    gpu_vector_operations_double_t g_vecs_double(vec_size, CUBLAS);
    cpu_vector_operations_t c_vecs(vec_size, dot_prod_type);
    gpu_reduction_t reduction(vec_size);
    gpu_reduction_ogita_t reduciton_ogita(vec_size);
    threaded_reduction_t threaded_reduce(vec_size, -1, dot_prod_type);
    //testing a long double. CUMP has a minimum 64bits encoding.
    dot_product_cump_t reduction_cump(vec_size, 105); 


    unsigned int exact_bits = 1024;
    dot_exact_t dp_ref(exact_bits);
    sum_exact_t s_ref(exact_bits);
    dot_exact_double_t dp_double_ref(exact_bits);

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

    generate_vector_pair_t generator(&g_vecs, &g_vecs_double, &dp_double_ref);
    T cond_estimste = generator.generate(u1_d, u2_d, cond_number_);
    printf("condition estimate = %le\n", cond_estimste);
 
    

    T norm_u1 = g_vecs.norm(u1_d);
    T norm_u2 = g_vecs.norm(u2_d);
    g_vecs.use_high_precision();
    T norm_u1_o = g_vecs.norm(u1_d);
    T norm_u2_o = g_vecs.norm(u2_d);
    g_vecs.use_standard_precision();
    printf("||u1|| = %.24le, ||u2|| = %.24le \n", double(norm_u1), double(norm_u2) );
    printf("||u1||o= %.24le, ||u2||o= %.24le \n", double(norm_u1_o), double(norm_u2_o) );
    printf("  err1 = %.24le,   err2 = %.24le \n", double(std::abs(norm_u1_o - norm_u1)), double(std::abs(norm_u2_o - norm_u2)) );

// save to host 
    g_vecs.get(u1_d, u1_c);
    g_vecs.get(u2_d, u2_c);

    std::vector<float> wall_times;
    wall_times.reserve(rep);

    if(operation_type == 0 || operation_type == 2)
    {
        printf("========================= dot =========================\n");    
        
        T dot_cump = 0;
        reduction_cump.set_arrays(u1_c, u2_c);
        reduction_cump.dot_benchmark(rep);
        dot_cump =reduction_cump.dot();
        std::vector<float> wall_time_cump = reduction_cump.get_repeated_execution_time_milliseconds();
        wall_times = without_hi_lo<float>(wall_time_cump);
        auto min_max = std::minmax_element(begin(wall_times), end(wall_times));
        auto average = accumulate( wall_times.begin(), wall_times.end(), 0.0)/wall_times.size();
        wall_times.clear();
        
        printf("dot_CU= %.24le, {min, max, ave}time = %f, %f, %f ms\n", double(dot_cump), *min_max.first, *min_max.second, average);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        g_vecs.use_standard_precision();
        T dot_prod_1 = 0;
        for(int r = 0;r<rep;r++)
        {
            cudaEventRecord(start); 
            dot_prod_1 = g_vecs.scalar_prod(u1_d, u2_d);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            wall_times.push_back(milliseconds);
        }
        wall_times = without_hi_lo<float>(wall_times);
        min_max = std::minmax_element(begin(wall_times), end(wall_times));
        average = accumulate( wall_times.begin(), wall_times.end(), 0.0)/wall_times.size();
        wall_times.clear();  

        printf("dot_L = %.24le, {min, max, ave}time = %f, %f, %f ms\n", double(dot_prod_1),  *min_max.first, *min_max.second, average);
        
        
        auto start_ch = std::chrono::steady_clock::now();
        T dot_prod_3 = reduction.dot(u1_d, u2_d);
        auto finish_ch = std::chrono::steady_clock::now();
        T elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();

        printf("dot_G = %.24le, time_wall = %lf ms\n", double(dot_prod_3), elapsed_mseconds);

        start_ch = std::chrono::steady_clock::now();
        T dot_prod_C_th = threaded_reduce.dot(u1_c, u2_c);
        finish_ch = std::chrono::steady_clock::now();
        elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
        printf("dot_Ct= %.24le, time_wall = %lf ms\n", double(dot_prod_C_th), double(elapsed_mseconds) );
        
        T dot_prod_ogita_G = 0;

        for(int r = 0;r<rep;r++)
        {
            start_ch = std::chrono::steady_clock::now();
            dot_prod_ogita_G = reduciton_ogita.dot(u1_d, u2_d);
            cudaDeviceSynchronize();
            finish_ch = std::chrono::steady_clock::now();
            elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();  
            wall_times.push_back(elapsed_mseconds);      
        }
        wall_times = without_hi_lo<float>(wall_times);
        min_max = std::minmax_element(begin(wall_times), end(wall_times));
        average = accumulate( wall_times.begin(), wall_times.end(), 0.0)/wall_times.size();
        wall_times.clear();  

        printf("dot_OG= %.24le, {min, max, ave}time = %f, %f, %f ms\n", double(dot_prod_ogita_G),  *min_max.first, *min_max.second, average);
        

        if(use_ref)
        {

            if(host_or_device == 'g')
            {
                dp_ref.use_gpu(vec_size);
            }
            else if(host_or_device == 'c')
            {
                dp_ref.use_cpu();
            }
            dp_ref.set_arrays(vec_size, u1_c, u2_c);
            

            start_ch = std::chrono::steady_clock::now();
            T dot_prod_2 = c_vecs.scalar_prod(u1_c, u2_c);
            finish_ch = std::chrono::steady_clock::now();
            elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
            printf("dot_C = %.24le, time_wall = %lf ms\n", double(dot_prod_2), double(elapsed_mseconds) );
            
            T ref_exact = dp_ref.dot_exact();
            // T ref_exact_gpu = dp_ref.dot_exact();
            // dp_ref.use_cpu();
            // T ref_exact = dp_ref.dot_exact();
            // std::cout << std::scientific << "diff: " << ref_exact_gpu - ref_exact << std::endl;
            T error_dot_L = dp_ref.get_error_relative(dot_prod_1);
            T error_dot_G = dp_ref.get_error_relative(dot_prod_3);
            T error_dot_C = dp_ref.get_error_relative(dot_prod_2);
            T error_dot_C_th = dp_ref.get_error_relative(dot_prod_C_th);
            T error_dot_G_ogita = dp_ref.get_error_relative(dot_prod_ogita_G);
            T error_dot_CUMP = dp_ref.get_error_relative(dot_cump);
            

            printf("ref   = ");
            dp_ref.print_res();       
            printf("ref   = %.24le\n", double(ref_exact)); 
            printf("mantisa: \033[0;31mX.123456789123456789\033[0m\n");
            printf("err_L = %.24le; err_G = %.24le; err_Ct = %.24le; err_C = %.24le; err_G_ogita = %.24le; err_CUMP = %.24le \n", double(error_dot_L), double(error_dot_G), double(error_dot_C_th), double(error_dot_C), double(error_dot_G_ogita), double(error_dot_CUMP) );
        }
    }
    if(operation_type == 0 || operation_type == 1)
    {
        printf("========================= sum =========================\n");
        //sum reduction
        T asum_L = 0;
        for(int r = 0;r<rep;r++)
        {
            auto start_ch = std::chrono::steady_clock::now();
            asum_L = g_vecs.absolute_sum(u2_d);
            auto finish_ch = std::chrono::steady_clock::now();
            auto elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
            wall_times.push_back(elapsed_mseconds);
        }
        wall_times = without_hi_lo<float>(wall_times);
        auto min_max = std::minmax_element(begin(wall_times), end(wall_times));
        auto average = accumulate( wall_times.begin(), wall_times.end(), 0.0)/wall_times.size();
        wall_times.clear();

        printf("asum_L = %.24le, {min, max, ave}time = %f, %f, %f ms\n", double(asum_L),  *min_max.first, *min_max.second, average);

        auto start_ch = std::chrono::steady_clock::now();
        T asum_G = reduction.asum(u2_d);
        auto finish_ch = std::chrono::steady_clock::now();
        auto elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
        printf("asum_G = %.24le, time_wall = %lf ms\n", double(asum_G), elapsed_mseconds);
        start_ch = std::chrono::steady_clock::now();
        T sum_G = reduction.sum(u2_d);
        finish_ch = std::chrono::steady_clock::now();
        elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
        if(sum_G<T(0.0))
            printf("sum_G  =%.24le, time_wall = %lf ms\n", double(sum_G), elapsed_mseconds);    
        else
            printf("sum_G  = %.24le, time_wall = %lf ms\n", double(sum_G), elapsed_mseconds);    

        T sum_ogita_G = 0.0;

        for(int r = 0;r<rep;r++)
        {
            start_ch = std::chrono::steady_clock::now();
            sum_ogita_G = reduciton_ogita.sum(u2_d);
            finish_ch = std::chrono::steady_clock::now();
            elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
            wall_times.push_back(elapsed_mseconds);
        }            
        wall_times = without_hi_lo<float>(wall_times);
        min_max = std::minmax_element(begin(wall_times), end(wall_times));
        average = accumulate( wall_times.begin(), wall_times.end(), 0.0)/wall_times.size();
        wall_times.clear();

        if(sum_ogita_G<T(0.0))
            printf("sum_OgG=%.24le, {min, max, ave}time = %f, %f, %f ms\n", double(sum_ogita_G), *min_max.first, *min_max.second, average);
        else
            printf("sum_OgG= %.24le, {min, max, ave}time = %f, %f, %f ms\n", double(sum_ogita_G), *min_max.first, *min_max.second, average);


        printf("d_sum  = %.24le \n", std::abs<T>(sum_ogita_G - sum_G) );

        start_ch = std::chrono::steady_clock::now();
        threaded_reduce.use_standard_precision();
        T sum_C_th = threaded_reduce.sum(u2_c);
        finish_ch = std::chrono::steady_clock::now();
        elapsed_mseconds = std::chrono::duration<double, std::milli>(finish_ch - start_ch).count();
        if(sum_C_th<T(0.0))
            printf("sum_ThC=%.24le, time_wall = %lf ms\n", double(sum_C_th), elapsed_mseconds);
        else
            printf("sum_ThC= %.24le, time_wall = %lf ms\n", double(sum_C_th), elapsed_mseconds);

        if(use_ref)
        {

            s_ref.set_array(vec_size, u2_c);
            
            start_ch = std::chrono::steady_clock::now();
            threaded_reduce.use_high_precision();
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
    }
    
    g_vecs.free_vector(u1_d); g_vecs.free_vector(u2_d); g_vecs.free_vector(u3_d); 
    c_vecs.free_vector(u1_c); c_vecs.free_vector(u2_c);

    delete CUBLAS;

    return 0;
}
