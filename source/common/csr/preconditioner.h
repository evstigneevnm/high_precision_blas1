#ifndef __CSR__PRECONDITIONER_H__
#define __CSR__PRECONDITIONER_H__

#include <stdexcept>


namespace csr
{
template<class VectorOperations, class LinearOperator, class Matrix>
struct preconditioner
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    preconditioner(const VectorOperations* vec_ops_):
    vec_ops(vec_ops_)
    {
        vec_ops->init_vector(y); vec_ops->start_use_vector(y);
    }
    ~preconditioner()
    {
        vec_ops->stop_use_vector(y); vec_ops->free_vector(y);
    }

    void set_operator(const LinearOperator* lin_op_)
    {
        lin_op = lin_op_;
        operator_set = true;
    }

    void set_matrix(const Matrix* mat_)
    {
        mat = mat_;
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
            mat->axpy(1.0, y, 0.0, x);
        }
    }
private:
    mutable T_vec y = nullptr;
    bool operator_set = false;
    bool matrix_set = false;
    const VectorOperations* vec_ops;
    const LinearOperator* lin_op;
    const Matrix* mat;    

};
}
#endif