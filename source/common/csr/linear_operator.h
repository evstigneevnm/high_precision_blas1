#ifndef __CSR__LINEAR_OPERATOR_H__
#define __CSR__LINEAR_OPERATOR_H__

namespace csr
{
template <class VectorOperations, class Matrix>
struct linear_operator
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    linear_operator(const VectorOperations* vec_ops_, Matrix* mat_):
        vec_ops(vec_ops_), mat(mat_)
    {
    }
    void apply(const T_vec& x, T_vec& f)const
    {
        mat->axpy(1.0, (T_vec&)x, 0.0, f);
    }
private:
    const VectorOperations* vec_ops;
    const Matrix* mat;       

};
}

#endif