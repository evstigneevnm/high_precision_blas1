#include <utils/cuda_support.h>
#include <utils/cusparse_safe_call.h>
#include <thrust/complex.h>
#include <common/csr/matrix_market_reader.h>
#include <cusparse.h>
#include <type_traits>
#include <iomanip>

template<class T, class Ord>
void save_device_vec_2_file(Ord sz_, T* vec_device, std::string f_name)
{
    T* vec;
    vec = host_allocate<T>(sz_);
    device_2_host_cpy<T>(vec, vec_device, sz_);
    
    std::ofstream check_file(f_name);
    if( !check_file.is_open() )
    {
        throw std::runtime_error("failed to open file " + f_name);
    }
    for(Ord j=0; j<sz_-1; j++)
    {
        if(!( check_file << std::scientific << std::setprecision(16) << vec[j] << std::endl))
        {
            throw std::runtime_error("failed to write to file " + f_name + " at " + std::to_string(j));
        }
    }
    if(!(check_file << std::scientific << std::setprecision(16) << vec[sz_-1]))
    {
        throw std::runtime_error("failed to write to file " + f_name + " at " + std::to_string(sz_-1));
    }
    check_file.close();
    host_deallocate(vec);
}


template<class Ord>
cusparseIndexType_t cusparse_ordinal()
{
    if((std::is_same<Ord, std::int32_t>::value)||(std::is_same<Ord, std::uint32_t>::value))
    {
        // std::cout << "index 32\n";
        return CUSPARSE_INDEX_32I;
    }
    if((std::is_same<Ord, std::int64_t>::value)||(std::is_same<Ord, std::uint64_t>::value))
    {
        // std::cout << "index 64\n";
        return CUSPARSE_INDEX_64I;
    }
    return CUSPARSE_INDEX_64I;
}
    
template<class T>
cudaDataType cusparse_real()
{
    if(std::is_same<T, float>::value)
    {
        // std::cout << "float\n";
        return CUDA_R_32F;
    }
    if(std::is_same<T, double>::value)
    {
        // std::cout << "double\n";
        return CUDA_R_64F;
    }
    if(std::is_same<T, thrust::complex<float> >::value)
    {
        // std::cout << "complex float\n";
        return CUDA_C_32F;
    }   
    if(std::is_same<T, thrust::complex<double> >::value)
    {
        // std::cout << "complex double\n";
        return CUDA_C_64F;
    } 
}

template<class T, class Ord>
struct mat_struct
{
    T* data = nullptr;
    Ord* row_data = nullptr;
    Ord* col_data = nullptr;

    Ord row_dim;
    Ord col_dim;
    Ord nnz;
    Ord col_csr;
    Ord row_csr;

};

template<class T, class Ord = size_t>
void axpy(mat_struct<T, Ord> A, T alpha_, T* x_, T beta_, T* y_, bool transpose_ = false)
{
    cusparseDnVecDescr_t vecX, vecY;
    cusparseSpMatDescr_t mat;
    cusparseOperation_t is_transpose;
    cusparseHandle_t handle = 0;
    CUSPARSE_SAFE_CALL( cusparseCreate(&handle) );
    
    if(transpose_)
    {
        is_transpose = CUSPARSE_OPERATION_TRANSPOSE;
    }
    else
    {
        is_transpose = CUSPARSE_OPERATION_NON_TRANSPOSE;
    }

    CUSPARSE_SAFE_CALL
    (
        cusparseCreateDnVec(&vecX, A.col_csr, x_, cusparse_real<T>() )
    );
    CUSPARSE_SAFE_CALL
    (
        cusparseCreateDnVec(&vecY, A.row_csr, y_, cusparse_real<T>() )
    ); 
    
    CUSPARSE_SAFE_CALL
    (
        cusparseCreateCsr
        (
            &mat,
            A.row_csr, A.col_csr, A.nnz,
            A.row_data, A.col_data, A.data,
            cusparse_ordinal<Ord>(), cusparse_ordinal<Ord>(), CUSPARSE_INDEX_BASE_ZERO,
            cusparse_real<T>()
        )
    );

    size_t bufferSize = 0;       
    CUSPARSE_SAFE_CALL
    ( 
        cusparseSpMV_bufferSize
        (
            handle, 
            is_transpose,
            &alpha_, mat, vecX, &beta_, vecY, cusparse_real<T>(),
            CUSPARSE_MV_ALG_DEFAULT, &bufferSize) 
    );
    void* dBuffer = NULL;
    CUDA_SAFE_CALL
    (
        cudaMalloc(&dBuffer, bufferSize)
    );
    CUSPARSE_SAFE_CALL
    ( 
        cusparseSpMV
        (
            handle, 
            is_transpose,
            &alpha_, mat, vecX, &beta_, vecY, cusparse_real<T>(),
            CUSPARSE_MV_ALG_DEFAULT, dBuffer) 
    );
    CUDA_SAFE_CALL
    (
        cudaFree(dBuffer)
    );

    CUSPARSE_SAFE_CALL( cusparseDestroySpMat(mat) );
    CUSPARSE_SAFE_CALL( cusparseDestroyDnVec(vecX) );
    CUSPARSE_SAFE_CALL( cusparseDestroyDnVec(vecY) );
    CUSPARSE_SAFE_CALL( cusparseDestroy(handle) );
}


int main(int argc, char const *argv[])
{
    using T = TYPE;
    using Ord = size_t;
    using mm_reader_t = csr::matrix_market_reader<T, Ord>;
    using mat_struct_t = mat_struct<T, Ord>;

    if(argc!=3)
    {
        std::cout << "usage: " << argv[0] << " ID file_name.mtx" << std::endl;
        std::cout << "where ID is the gpu device pci bus id or -1 for explicit selection." << std::endl;
        return(0);
    }

    std::string file_name(argv[2]);
    int gpu_pci_id = std::atoi(argv[1]);
    init_cuda(gpu_pci_id);

    mm_reader_t reader(true);
    reader.read_file(file_name);

    size_t row_, col_;
    mat_struct_t Ah;
    mat_struct_t A;

    reader.get_matrix_dim(A.row_dim, A.col_dim);
    reader.allocate_set_csr_pointers(A.row_csr, A.col_csr, A.nnz, Ah.row_data, Ah.col_data, Ah.data);
    row_ = A.row_dim;
    col_ = A.col_dim;

    A.data = device_allocate<T>(A.nnz);
    A.row_data = device_allocate<Ord>(A.row_csr);
    A.col_data = device_allocate<Ord>(A.col_csr);
    host_2_device_cpy(A.data, Ah.data, A.nnz);
    host_2_device_cpy(A.row_data, Ah.row_data, A.row_csr);
    host_2_device_cpy(A.col_data, Ah.col_data, A.col_csr);    
    free(Ah.data);
    free(Ah.row_data);
    free(Ah.col_data);


    T* xh; T* yh; 
    T* x; T* y;

    xh = (T*)malloc(sizeof(T)*col_);
    yh = (T*)malloc(sizeof(T)*row_);
    for(int j=0;j<col_;j++)
    {
        xh[j] = 1.0;
    }
    for(int j=0;j<row_;j++)
    {
        yh[j] = 2.0;
    }  

    x = device_allocate<T>(col_);
    y = device_allocate<T>(row_);
    host_2_device_cpy(x, xh, col_);
    host_2_device_cpy(y, yh, row_);

    save_device_vec_2_file<T, Ord>(row_, y, "y.dat");
    save_device_vec_2_file<T, Ord>(col_, x, "x.dat");
    
    axpy<T, Ord>(A, 1.0, x, 1.0, y);

    save_device_vec_2_file<T, Ord>(row_, y, "res.dat");

    

    cudaFree(A.data);
    cudaFree(A.row_data);
    cudaFree(A.col_data);
    device_deallocate(y); device_deallocate(x);
    free(yh); free(xh);

    return 0;
}