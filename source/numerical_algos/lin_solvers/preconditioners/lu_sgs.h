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
#ifndef __SCFD_LU_SGS_H__
#define __SCFD_LU_SGS_H__

#include <utils/logged_obj_base.h>

namespace numerical_algos
{
namespace lin_solvers 
{

/*struct operator_operations
{
    typedef scalar_type;
    typedef operator_type;

    void start_use_operator(operator_type &op);
    void stop_use_operator(operator_type &op);

    void start_use_operator_upper(operator_type &op);
    void stop_use_operator_upper(operator_type &op);

    void start_use_operator_lower(operator_type &op);
    void stop_use_operator_lower(operator_type &op);

    void start_use_operator_inverted_upper(operator_type &op);
    void stop_use_operator_inverted_upper(operator_type &op);

    void start_use_operator_inverted_lower(operator_type &op);
    void stop_use_operator_inverted_lower(operator_type &op);

    void add_mul_scalar(operator_type &op, scalar_type scalar);
};*/

template<class LinearOperator,class VectorOperations,class Log>
class lu_sgs : public utils::logged_obj_base<Log>
{
public:
    typedef LinearOperator                          operator_type;
    typedef VectorOperations                        vector_operations_type;
    typedef typename VectorOperations::vector_type  vector_type;

private:
    typedef vectors_arr_wrap_static<VectorOperations,1>         bufs_arr_t;
    typedef typename bufs_arr_t::vectors_arr_use_wrap_type      bufs_arr_use_wrap_t;

    const vector_operations_type    *vec_ops_;
    const LinearOperator            *op_;

    mutable bufs_arr_t              bufs;
    vector_type                     &tmp;

public:
    lu_sgs(const vector_operations_type *vec_ops, 
             Log *log = NULL, int obj_log_lev = 0) : 
        utils::logged_obj_base<Log>(log, obj_log_lev, "lu_sgs::"),
        vec_ops_(vec_ops), op_(NULL), bufs(vec_ops), tmp(bufs[0])
    {
        bufs.init();
    }

    void set_operator(const LinearOperator *op)
    {
        op_ = op;
    }

    void apply(vector_type &x)const
    {
        assert(op_ != NULL);

        bufs_arr_use_wrap_t     use_wrap(bufs);
        use_wrap.start_use_all();

        op_->apply_inverted_lower(x, tmp);
        op_->apply_inverted_upper(tmp, x);
    }
};

}
}

#endif