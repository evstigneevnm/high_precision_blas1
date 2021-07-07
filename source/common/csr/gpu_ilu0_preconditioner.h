#ifndef __CSR__GPU_ILU0_PRECONDITIONER_H__
#define __CSR__GPU_ILU0_PRECONDITIONER_H__

#include <stdexcept>


namespace csr
{
template<class VectorOperations, class LinearOperator, class Matrix>
struct gpu_ilu0_preconditioner
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    gpu_ilu0_preconditioner(const VectorOperations* vec_ops_):
    vec_ops(vec_ops_)
    {
        vec_ops->init_vector(y); vec_ops->start_use_vector(y);
    }
    ~gpu_ilu0_preconditioner()
    {
        if(mat != nullptr) delete mat;
        vec_ops->stop_use_vector(y); vec_ops->free_vector(y);
    }

    void set_operator(const LinearOperator* lin_op_)
    {
        lin_op = lin_op_;
        operator_set = true;
    }
    void set_matrix(const Matrix matP_)
    {
        mat = new Matrix(matP_);
        matrix_set = true;
    }

    void apply(T_vec& x)const
    {
        if(!matrix_set)
        {
            //throw std::logic_error("csr::preconditioner: operator not set. Use set_operator(Matrix).");
            
        }
        else
        {
            vec_ops->assign(x, y);
            
        }
    }
private:
    mutable T_vec y = nullptr;
    bool operator_set = false;
    bool matrix_set = false;
    const VectorOperations* vec_ops;
    const LinearOperator* lin_op;
    Matrix* mat = nullptr;
    

};
}
#endif