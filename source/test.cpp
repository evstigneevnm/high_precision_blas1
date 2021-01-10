#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <random_adapter.hpp>
#include <sums.hpp>
#include <dot_product.hpp>



int main(int argc, char const *argv[])
{
    
    typedef TYPE T;
    using S_t = sums<T>;
    using D_t = dot_product<T>;
    // mpz_class a, b, c;
    unsigned int max_bits = 2048;

    if(argc != 2)
    {
        std::printf("Usage: %s N\n N - array size\n", argv[0]);
        return 0;
    }
    size_t N = std::atoi(argv[1]);
    T* array = new T[N];

    T* X = new T[N];
    T* Y = new T[N];

    random_adapter<T> RA;
    RA.set_uniform_distribution(-500.0, 500.0);

    for(size_t j=0;j<N;j++)
    {
        array[j] = RA.get_uniform();
        X[j] = RA.get_uniform();
        Y[j] = RA.get_uniform();
    }    


    S_t S_cpu(max_bits, false);
    D_t D_cpu(max_bits, false);

    S_cpu.set_array(N, array);
    D_cpu.set_arrays(N, X, Y);

    T sum_exact = S_cpu.sum_exact();
    T sum_kahan = S_cpu.sum_kahan();
    T sum_naive = S_cpu.sum_naive();
    T sum_rump = S_cpu.sum_rump();

    T dot_exact = D_cpu.dot_exact();
    T dot_naive = D_cpu.dot_naive();
    T dot_fma = D_cpu.dot_fma();
    T dot_ogita = D_cpu.dot_ogita();

    S_cpu.error_check();
    std::printf("array sum:\n");
    std::printf("sum exact = %.52Le \n", (long double)sum_exact);
    std::printf("sum naive = %.52Le \n", (long double)sum_naive);
    std::printf("sum Kahan = %.52Le \n", (long double)sum_kahan);
    std::printf("sum rump  = %.52Le \n", (long double)sum_rump);

    std::printf("%.22le %.22le\n %.22le %.22le\n %.22le %.22le\n\n", (double)S_cpu.error_naive(), (double) (sum_naive - sum_exact),(double)S_cpu.error_kahan(), (double) (sum_kahan - sum_exact), (double)S_cpu.error_rump(), (double) (sum_rump - sum_exact) );
    std::printf("arrays dot product:\n");
    std::printf("dot exact = %.52Le \n", (long double)dot_exact);
    std::printf("dot naive = %.52Le \n", (long double)dot_naive);
    std::printf("dot fma = %.52Le \n", (long double)dot_fma);
    std::printf("dot ogita  = %.52Le \n", (long double)dot_ogita);

    std::printf("%.22le %.22le\n %.22le %.22le\n %.22le %.22le\n\n", (double)D_cpu.error_naive(), (double) (dot_naive - dot_exact), (double)D_cpu.error_fma(),(double) (dot_fma - dot_exact), (double)D_cpu.error_ogita(),(double) (dot_ogita - dot_exact) );

    delete [] array;
    delete [] X;
    delete [] Y;
    return 0;
}