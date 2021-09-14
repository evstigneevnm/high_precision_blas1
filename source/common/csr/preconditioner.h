/*
 * MIT License
 *
 * Copyright (c) 2020 Evstigneev Nikolay Mikhaylovitch <evstigneevnm@ya.ru>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
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

    void set_matrix(const Matrix* matL_, const Matrix* matU_)
    {
        matU = matU_;
        matL = matL_;
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
            matL->triangular_solve_lower(x, y);
            matU->triangular_solve_upper(y, x);
            
        }
    }
private:
    mutable T_vec y = nullptr;
    bool operator_set = false;
    bool matrix_set = false;
    const VectorOperations* vec_ops;
    const LinearOperator* lin_op;
    const Matrix* matL;
    const Matrix* matU;    

};
}
#endif