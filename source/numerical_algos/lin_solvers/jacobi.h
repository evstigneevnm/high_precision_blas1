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

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_JACOBI_H__
#define __SCFD_JACOBI_H__

#include <numerical_algos/detail/vectors_arr_wrap_static.h>
#include "detail/monitor_call_wrap.h"
#include "iter_solver_base.h"

namespace numerical_algos
{
namespace lin_solvers 
{

using numerical_algos::detail::vectors_arr_wrap_static;

//demands for template parameters:
//VectorOperations fits VectorOperations concept (see CONCEPTS.txt)
//LinearOperator and Preconditioner fit LinearOperator concept (see CONCEPTS.txt)
//VectorOperations, LinearOperator and Preconditioner have same vector_type

//Monitor concept:
//TODO

template<class LinearOperator,class Preconditioner,
         class VectorOperations,class Monitor,class Log>
class jacobi : public iter_solver_base<LinearOperator,Preconditioner,
                                       VectorOperations,Monitor,Log>
{
public:
    typedef typename VectorOperations::scalar_type  scalar_type;
    typedef typename VectorOperations::vector_type  vector_type;
    typedef LinearOperator                          linear_operator_type;
    typedef Preconditioner                          preconditioner_type;
    typedef VectorOperations                        vector_operations_type;
    typedef Monitor                                 monitor_type;
    typedef Log                                     log_type;

private:
    typedef scalar_type                                         T;
    typedef utils::logged_obj_base<Log>                         logged_obj_t;
    typedef iter_solver_base<LinearOperator,Preconditioner,
                             VectorOperations,Monitor,Log>      parent_t;
    typedef vectors_arr_wrap_static<VectorOperations,1>         bufs_arr_t;
    typedef typename bufs_arr_t::vectors_arr_use_wrap_type      bufs_arr_use_wrap_t;
    typedef detail::monitor_call_wrap<VectorOperations,
                                      Monitor>                  monitor_call_wrap_t;


    mutable bufs_arr_t   bufs;
    vector_type          &ri;

protected:
    using parent_t::monitor_;
    using parent_t::vec_ops_;
    using parent_t::prec_;

public:
    jacobi(const vector_operations_type *vec_ops, 
           Log *log = NULL, int obj_log_lev = 0) : 
        parent_t(vec_ops, log, obj_log_lev, "jacobi::"),
        bufs(vec_ops), ri(bufs[0])
    {
        bufs.init();
    }

    virtual bool    solve(const linear_operator_type &A, const vector_type &b, 
                          vector_type &x)const
    {
        if (prec_ != NULL) prec_->set_operator(&A);
        
        bufs_arr_use_wrap_t     use_wrap(bufs);
        use_wrap.start_use_all();

        monitor_call_wrap_t     monitor_wrap(monitor_);
        monitor_wrap.start(b);

        //ri := b - A*x0;
        A.apply(x, ri);                                 //ri := A*x0
        vec_ops_->add_mul(T(1.f), b, -T(1.f), ri);      //ri := -ri + b = -A*x0 + b

        while (!monitor_.check_finished(x, ri)) {
            if (prec_ != NULL) prec_->apply(ri);            //r{i-1} := P*r{i-1}

            //xi := x{i-1} + r{i-1}
            vec_ops_->add_mul(T(1.f), ri, T(1.f), x);
            
            //ri := b - A*xi;
            A.apply(x, ri);                             //ri := A*xi
            vec_ops_->add_mul(T(1.f), b, -T(1.f), ri);  //ri := -ri + b = -A*xi + b

            ++monitor_;
        }
        
        if (monitor_.out_min_resid_norm()) vec_ops_->assign(monitor_.min_resid_norm_x(), x);

        return monitor_.converged();
    }
};

}
}

#endif
