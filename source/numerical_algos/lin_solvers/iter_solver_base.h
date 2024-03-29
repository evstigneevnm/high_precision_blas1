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

#ifndef __SCFD_ITER_SOLVER_BASE_H__
#define __SCFD_ITER_SOLVER_BASE_H__

#include <utils/logged_obj_base.h>

namespace numerical_algos
{
namespace lin_solvers 
{


template<class LinearOperator,class Preconditioner,
         class VectorOperations,class Monitor,class Log>
class iter_solver_base : public utils::logged_obj_base<Log>
{
public:
    typedef typename VectorOperations::scalar_type  scalar_type;
    typedef typename VectorOperations::vector_type  vector_type;
    typedef LinearOperator                          linear_operator_type;
    typedef Preconditioner                          preconditioner_type;
    typedef VectorOperations                        vector_operations_type;
    typedef Monitor                                 monitor_type;
    typedef Log                                     log_type;

protected:
    mutable monitor_type           monitor_;
    const vector_operations_type   *vec_ops_;
    //NOTE if prec_ != NULL and own_prec_ is true, solver OWNS prec, so solver must delete it
    preconditioner_type            *prec_;
    bool                           own_prec_;
public:
    iter_solver_base(const vector_operations_type *vec_ops, 
                     Log *log, int obj_log_lev, const std::string &log_msg_prefix) :
        utils::logged_obj_base<Log>(log, obj_log_lev, log_msg_prefix), 
        monitor_(*vec_ops, log, obj_log_lev),
        vec_ops_(vec_ops), prec_(NULL) 
    {
        monitor_.set_log_msg_prefix(log_msg_prefix + monitor_.get_log_msg_prefix());
    }

    Monitor         &monitor() { return monitor_; }
    const Monitor   &monitor()const { return monitor_; }

    void            set_preconditioner(preconditioner_type *prec, bool own_prec = false) 
    { 
        prec_ = prec; 
        own_prec_ = own_prec;
    }

    virtual bool    solve(const linear_operator_type &A, const vector_type &b, 
                          vector_type &x)const = 0;

    virtual ~iter_solver_base()
    {
        if ((prec_ != NULL) && own_prec_) delete prec_;
    }
};

}
}

#endif

