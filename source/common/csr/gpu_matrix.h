#ifndef __CSR__GPU_MATRIX_H__
#define __CSR__GPU_MATRIX_H__


#include <common/csr/matrix.h>
#include <common/csr/gpu_vectors_ordinal.h>
#include <utils/cuda_safe_call.h>
#include <utils/cusparse_safe_call.h>
#include <utils/cuda_support.h>
#include <library_types.h>
#include <cusparse.h>
#include <thrust/complex.h>
#include <type_traits>

namespace csr
{

template<class VectorOperations, class CudaBlas, class Ordinal = size_t>
class gpu_matrix: public matrix<VectorOperations, gpu_vectors_ordinal<Ordinal>, CudaBlas>
{
private:
    using parent_t = matrix<VectorOperations, gpu_vectors_ordinal<Ordinal>, CudaBlas>;
    using T = typename parent_t::T;
    using T_vec = typename parent_t::T_vec;
    using I = typename gpu_vectors_ordinal<Ordinal>::scalar_type;
    using I_vec = typename gpu_vectors_ordinal<Ordinal>::vector_type;


    cusparseIndexType_t cusparse_ordinal()const
    {
        if((std::is_same<Ordinal, std::int32_t>::value)||(std::is_same<Ordinal, std::uint32_t>::value))
        {
            
            return CUSPARSE_INDEX_32I;
        }
        if((std::is_same<Ordinal, std::int64_t>::value)||(std::is_same<Ordinal, std::uint64_t>::value))
        {
            
            return CUSPARSE_INDEX_64I;
        }
        return CUSPARSE_INDEX_64I;
    }
        

    cudaDataType cusparse_real()const
    {
        if(std::is_same<T, float>::value)
        {

            return CUDA_R_32F;
        }
        if(std::is_same<T, double>::value)
        {
            
            return CUDA_R_64F;
        }
        if(std::is_same<T, thrust::complex<float> >::value)
        {
            
            return CUDA_C_32F;
        }   
        if(std::is_same<T, thrust::complex<double> >::value)
        {
            
            return CUDA_C_64F;
        } 
    }

public:
    
    gpu_matrix(CudaBlas* cublas_): parent_t(cublas_)
    {
        if(handle == 0)
        {
            cusparseCreate(&handle);
        }        
    }

    gpu_matrix(int nnz_, int size_col_, int size_row_, CudaBlas* cublas_): parent_t(nnz_, size_col_, size_row_, cublas_)
    {
        if(handle == 0)
        {
            CUSPARSE_SAFE_CALL( cusparseCreate(&handle) );
        }
    }    
    ~gpu_matrix()
    {
        if(mat != 0)
        {
            cusparseDestroySpMat(mat);
        }
        if(handle != 0)
        {
            cusparseDestroy(handle);
        }

    }
    
    void set(const T_vec data_, const I_vec col_ind_, const I_vec row_ptr_)
    {
        try
        {
            parent_t::set(data_, col_ind_, row_ptr_);
        }
        catch(std::exception)
        {
            throw(std::runtime_error("csr::gpu_matrix: failed to set the gpu_matrix."));
        }
        
        init_cusparse();

    }


    /**
     * @brief      { y <- \alphaAx + \beta y }
     *
     * @param[in]  alpha_      basic type
     * @param[in]  x_          { basic type array }
     * @param[in]  beta_       basic type
     * @param      y_          { basic type array }
     * @param[in]  transpose_  true for transposed operation of A
     */
    void axpy(const T& alpha_, const T_vec& x_, const T& beta_, T_vec& y_, bool transpose_ = false)const
    {
        cusparseDnVecDescr_t vecX, vecY;
        cusparseOperation_t is_transpose;
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
            cusparseCreateDnVec(&vecX, parent_t::size_col, x_, cusparse_real())
        );
        CUSPARSE_SAFE_CALL
        (
            cusparseCreateDnVec(&vecY, parent_t::size_row, y_, cusparse_real())
        ); 
        size_t bufferSize = 0;       
        CUSPARSE_SAFE_CALL
        ( 
            cusparseSpMV_bufferSize
            (
                handle, 
                is_transpose,
                &alpha_, mat, vecX, &beta_, vecY, cusparse_real(),
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
                &alpha_, mat, vecX, &beta_, vecY, cusparse_real(),
                CUSPARSE_MV_ALG_DEFAULT, dBuffer) 
        );
        CUDA_SAFE_CALL
        (
            cudaFree(dBuffer)
        );

        CUSPARSE_SAFE_CALL( cusparseDestroyDnVec(vecX) );
        CUSPARSE_SAFE_CALL( cusparseDestroyDnVec(vecY) );

    }
private:
    cusparseHandle_t handle = 0;
    cusparseSpMatDescr_t mat = 0;

    void init_cusparse()
    {
        // cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_num_nnz, 
        //                               dA_csrOffsets, dA_columns, dA_values,
        //                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        //                               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)         
        
        CUSPARSE_SAFE_CALL
        (
            cusparseCreateCsr
            (
                &mat,
                parent_t::size_row, parent_t::size_col, parent_t::nnz,
                parent_t::row_ptr, parent_t::col_ind, parent_t::data,
                cusparse_ordinal(), cusparse_ordinal(), CUSPARSE_INDEX_BASE_ZERO,
                cusparse_real()
            )
        );        
    }


};
}
#endif