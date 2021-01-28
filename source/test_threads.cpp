/***
main program to test threaded and serial variations
*/

#include <iostream>
#include <chrono>
#include <common/threaded_dot_product.h>
#include <common/dot_product.h>
#include <dot_product_gmp.hpp>

typedef TYPE T;

int main(int argc, char const *argv[])
{
    if(argc != 4)
    {
        std::cout << argv[0] << " o N th; where: 'o' is the type of dot product method (0-naive, 1-ogita); 'N' is the vector size; 'th' is the number of threads." << std::endl;
        return 0;
    }

    int dot_prod_type = atoi(argv[1]);
    int vec_size = atoi(argv[2]);
    int n_threads = atoi(argv[3]);
    
    //Fill two vectors with some values 
    T val1 = T(7.0e-4);
    T val2 = T(7.0e-4);
    //std::vector<T> vec1_v(vec_size, val1), vec2_v(vec_size, val2);
    std::vector<T> vec1_v(vec_size), vec2_v(vec_size);

    T* vec1_d = new T[vec_size];
    T* vec2_d = new T[vec_size];

    for(int j = 0; j<vec_size; j++)
    {
        vec1_d[j] = val1*std::pow(1.0,(j));
        vec2_d[j] = val2*std::pow(1.0,(j));
        vec1_v[j] = val1*std::pow(1.0,(j));
        vec2_v[j] = val2*std::pow(1.0,(j));
    }

    threaded_dot_prod<T, std::vector<T> > dot_prod_v(vec_size, n_threads, dot_prod_type);
    threaded_dot_prod<T, T* > dot_prod_d(vec_size, n_threads, dot_prod_type);

    //Launch nr_threads threads:
    auto start = std::chrono::steady_clock::now();
    T res_v = dot_prod_v.execute(vec1_v, vec2_v);
    auto finish = std::chrono::steady_clock::now();
    T elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<T> >(finish - start).count();
    std::cout << "execution wall time for std::vector = " << elapsed_seconds << "sec." << std::endl;  

    start = std::chrono::steady_clock::now();
    T res_d = dot_prod_d.execute(vec1_d, vec2_d);
    finish = std::chrono::steady_clock::now();
    elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<T> >(finish - start).count();
    std::cout << "execution wall time for T* = " << elapsed_seconds << "sec." << std::endl;  

    std::cout.precision(24);
    std::cout << std::scientific << "res_v = " << res_v << std::endl << "res_d = " << res_d << std::endl;
    
    dot_product<T> dp_check(vec_size, 1);
    T check_d = dp_check.dot_ogita(vec1_d, vec2_d);
    T check_o = dp_check.dot_naive(vec1_d, vec2_d);

    dot_product_gmp<T, std::vector<T>> dp_ref(256);
    dp_ref.set_arrays(vec_size, vec1_v, vec2_v);
    T ref_exact = dp_ref.dot_exact();
    T error_exact = dp_ref.get_error_relative_T(res_d);


    std::cout.precision(24);
    std::cout << std::scientific << "ref   = " << ref_exact << " err = " << error_exact << std::endl;
    std::cout << std::scientific << "chk_d = " << check_d << std::endl;
    std::cout << std::scientific << "chk_o = " << check_o << std::endl;
    std::cout << std::scientific << "mantisa " << "_.123456\033[1;31m7\033[0m8912345\033[1;31m6\033[0m789" << std::endl;
    std::cout << std::scientific << "least>1 " << "1.0000000000000002" << std::endl;
    //std::cout << "asd\033[1;31mbold red text\033[0m\n" << std::endl;
    //https://stackoverflow.com/questions/2616906/how-do-i-output-coloured-text-to-a-linux-terminal
    delete [] vec1_d;
    delete [] vec2_d;

    return 0;
}