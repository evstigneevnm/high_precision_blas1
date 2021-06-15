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
#include <high_prec/dot_product_gmp_complex.hpp>
#include <generate_vector_pair_complex.hpp>
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

template<>
std::string return_type_name(thrust::complex<float> some_var)
{
    return("complex_float");
}
template<>
std::string return_type_name(thrust::complex<double> some_var)
{
    return("complex_double");
}

int main(int argc, char const *argv[])
{
    
    using T = TYPE;
    using TC = thrust::complex<T>;
    using gpu_vector_operationsC_t = gpu_vector_operations<TC>;
    using cpu_vector_operationsC_t = cpu_vector_operations<TC>;
    using gpu_vector_operations_t = gpu_vector_operations<T>;
    using cpu_vector_operations_t = cpu_vector_operations<T>;
    using T_vec = gpu_vector_operations_t::vector_type;
    using TC_vec = gpu_vector_operationsC_t::vector_type;

    using dot_exact_t = dot_product_gmp_complex<T, T_vec>;
    using generate_vector_pair_t = generate_vector_pair_complex<gpu_vector_operationsC_t,gpu_vector_operations_t, dot_exact_t>;
    using threaded_reduction_t = threaded_reduction<TC, TC_vec>;
    using error_bounds_t = error_bounds<T>;    


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
    std::string type_name = return_type_name<TC>( TC(1.0) );
    init_cuda(gpu_pci_id);
    int dot_prod_type_initial = 0;
    dot_exact_t dp_ref(1024);

    cublas_wrap *CUBLAS_ref = new cublas_wrap(true);
    gpu_vector_operationsC_t g_vecCs(vec_size, CUBLAS_ref);
    cpu_vector_operationsC_t c_vecCs(vec_size, dot_prod_type_initial);
    gpu_vector_operations_t g_vecs(vec_size, CUBLAS_ref);
    cpu_vector_operations_t c_vecs(vec_size, dot_prod_type_initial);

    threaded_reduction_t threaded_reduce(vec_size, -1, dot_prod_type_initial);
    error_bounds_t err_bnd;

    std::cout << "Machine epsilon is " << err_bnd.macheps() << " " << std::endl;

    TC_vec u1_d; TC_vec u2_d; TC_vec u1_c; TC_vec u2_c;

    g_vecCs.init_vector(u1_d); g_vecCs.init_vector(u2_d);
    g_vecCs.start_use_vector(u1_d); g_vecCs.start_use_vector(u2_d);
    c_vecCs.init_vector(u1_c); c_vecCs.init_vector(u2_c); 
    c_vecCs.start_use_vector(u1_c); c_vecCs.start_use_vector(u2_c);
    printf("using vectors of size = %le\n", double(vec_size) );
    generate_vector_pair_t generator(&g_vecCs, &g_vecs, &dp_ref);
    

    T cond_number = T(1.0);

    std::string f_name = type_name + "_vec_size" + std::to_string(vec_size) + "_Cmax" + std::to_string(std::round(cond_number_max)) + ".dat";
    std::ofstream f(f_name.c_str(), std::ofstream::out);
    if (!f) throw std::runtime_error("error while opening file for writing: " + f_name);

    while( cond_number <= cond_number_max)
    {
        
        for(int le = 0; le < executions_step; le++)
        {
            auto cond_estimste = generator.generate(u1_d, u2_d, cond_number);
            std::cout << "condition estimate = re(" << cond_estimste.first << "), im(" << cond_estimste.second << ")" << std::endl;
            g_vecCs.use_standard_precision();
            TC dot_prod_BLAS = g_vecCs.scalar_prod(u1_d, u2_d);
            g_vecCs.use_high_precision();
            TC dot_prod_reduct_ogita = g_vecCs.scalar_prod(u1_d, u2_d);
            g_vecCs.use_standard_precision();
            std::cout << std::scientific << "dot_L  = " << dot_prod_BLAS << std::endl;   


            g_vecCs.get(u1_d, u1_c);
            g_vecCs.get(u2_d, u2_c);
            
            threaded_reduce.use_standard_precision();            
            TC dot_prod_th = threaded_reduce.dot(u1_c, u2_c);
            threaded_reduce.use_high_precision();
            TC dot_prod_th_H = threaded_reduce.dot(u1_c, u2_c);
            threaded_reduce.use_standard_precision();
            
            c_vecCs.use_standard_precision();
            TC dot_prod = c_vecCs.scalar_prod(u1_c, u2_c);
            c_vecCs.use_high_precision();
            TC dot_prod_H = c_vecCs.scalar_prod(u1_c, u2_c);            
            c_vecCs.use_standard_precision();

            std::cout << std::scientific << "dot_C  = " << dot_prod << std::endl;
            std::cout << std::scientific << "dot_Ct = " << dot_prod_th << std::endl;
            std::cout << std::scientific << "*dot_CH= " << dot_prod_H << std::endl;
            std::cout << std::scientific << "*dotCtH= " << dot_prod_th_H << std::endl;
            std::cout << std::scientific << "*dot_OG= " << dot_prod_reduct_ogita << std::endl;
           
/*            
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
            long double parallel24_bound_ = normalize_error<long double>(err_bnd.dot.real.base.pairwise_parallel);
            
            long double ogita_bound_ = normalize_error<long double>(err_bnd.dot.real.compensated.sequential );
            long double pairwise_comp_bound_ = normalize_error<long double>(err_bnd.dot.real.compensated.pairwise_parallel );
            long double parallel_comp_24_bound_ = normalize_error<long double>(err_bnd.dot.real.compensated.pairwise_parallel);

            // std::cout << "simple:" << simple_bound_ << " pairwise:" << pairwise_bound_ << " parallel24:" << parallel24_bound_ << " ogita:" << ogita_bound_ << " c_pairwise:" << pairwise_comp_bound_ << " c_parallel24:" << parallel_comp_24_bound_ << std::endl;
            

            printf("ref   = %.24le \n", double(ref_exact));        
            printf("mantisa:\033[0;31mX.123456789123456789\033[0m \n");
            printf("err_L = %.24le | %.24le \nerr_G = %.24le | %.24le \nerr_Ct = %.24le \nerr_C = %.24le | %.24le \n*err_CH= %.24le | %.24le \n*errCtH= %.24le | %.24le \n*err_GH= %.24le | %.24le\n", double(error_exact_L), double(pairwise_bound_), double(error_exact_G), double(pairwise_bound_), double(error_exact_C_th), double(error_exact_C), double(simple_bound_), double(error_exact_C_H), double(ogita_bound_), double(error_exact_C_th_H), double(parallel_comp_24_bound_), double(error_exact_ogita_G),  double(pairwise_comp_bound_));

            if ( !(f << cond_estimste  << " " << normalize_error(error_exact_L) << " " << normalize_error(error_exact_G) << " " << normalize_error(error_exact_ogita_G) << " " << normalize_error(error_exact_C) << " " <<  normalize_error(error_exact_C_H) << " " << normalize_error(error_exact_C_th) << " " << normalize_error(error_exact_C_th_H) << " " << ogita_bound_ << " " << parallel24_bound_ << " " << parallel_comp_24_bound_ << " " << pairwise_bound_ << " " << pairwise_comp_bound_ << " "<< simple_bound_ << std::endl ) )
            {
                throw std::runtime_error("error while writing to file: " + f_name);
            }
*/
        }

        cond_number *= cond_step_;
    }
    f.close();

    c_vecCs.free_vector(u1_c); c_vecCs.free_vector(u2_c);    
    g_vecCs.free_vector(u1_d); g_vecCs.free_vector(u2_d);
    

    delete CUBLAS_ref;

    return 0;
}