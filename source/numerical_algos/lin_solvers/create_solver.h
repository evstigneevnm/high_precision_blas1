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

#ifndef __SCFD_CREATE_SOLVER_H__
#define __SCFD_CREATE_SOLVER_H__

#include <string>
#include <boost/property_tree/ptree.hpp>
#include "create_monitor.h"
#include "solver_base.h"
#include "cgs.h"

namespace numerical_algos
{
namespace lin_solvers 
{

template<class LinearOperator,class Preconditioner,class VectorOperations,
         bool ManualVerifySize,class Log>
solver_base<LinearOperator,Preconditioner,VectorOperations,ManualVerifySize,default_monitor,Log>*
create_solver(const boost::property_tree::ptree &cfg, Log *log = NULL, int obj_log_lev = 0)
{
    std::string     lin_solver_type_name = cfg.get<std::string>("lin_solver_type_name"),

    if (lin_solver_type_name == "CGS") {
        return new cgs<LinearOperator,Preconditioner,VectorOperations,
                      ManualVerifySize,default_monitor,Log>(log, obj_log_lev);
    } else throw std::runtime_error("create_solver: unsupported lin_solver_type_name: " + 
                                    lin_solver_type_name);
    create_monitor(cfg.get_child("monitor"), log, obj_log_lev);
}


}
}

#endif
