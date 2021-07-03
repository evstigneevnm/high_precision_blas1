#include <thrust/complex.h>
#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <common/csr/gpu_matrix.h>
#include <common/csr/matrix_market_reader.h>

int main(int argc, char const *argv[])
{
    using T = TYPE;
    using TC = thrust::complex<T>;
    using gpu_vector_operations_t = gpu_vector_operations<T>;
    using T_vec = gpu_vector_operations_t::vector_type;
    using T_mat_t = csr::gpu_matrix<gpu_vector_operations_t, cublas_wrap>;
    using mm_reader_t = csr::matrix_market_reader<T>;

    if(argc!=3)
    {
        std::cout << "usage: " << argv[0] << " ID file_name.mtx" << std::endl;
        std::cout << "where ID is the gpu device pci bus id or -1 for explicit selection." << std::endl;
        return(0);
    }

    int gpu_pci_id = std::atoi(argv[1]);
    std::string file_name(argv[2]);
    init_cuda(gpu_pci_id);

    cublas_wrap cublas(true);
    T_mat_t mat(&cublas);
    // mat.init_set_from_file(file_name);
    mm_reader_t reader(true);
    reader.read_file(file_name);
    reader.set_csr_matrix<T_mat_t>(&mat);
    
    auto dims = mat.get_dim();
    int size_y = dims.rows;
    int size_x = dims.columns;
    gpu_vector_operations_t vec_ops_x(size_x, &cublas);
    gpu_vector_operations_t vec_ops_y(size_y, &cublas);
    T_vec x; T_vec y;
    vec_ops_x.init_vector(x); vec_ops_x.start_use_vector(x);
    vec_ops_y.init_vector(y); vec_ops_y.start_use_vector(y);

    vec_ops_x.assign_random(x); 
    vec_ops_y.assign_random(y);
    vec_ops_x.debug_view(x, "x.dat");
    vec_ops_y.debug_view(y, "y.dat");
    mat.axpy(1.0, x, 1.0, y);
    vec_ops_y.debug_view(y, "res.dat");


    vec_ops_y.stop_use_vector(y); vec_ops_y.free_vector(y);
    vec_ops_x.stop_use_vector(x); vec_ops_x.free_vector(x);

    return 0;
}