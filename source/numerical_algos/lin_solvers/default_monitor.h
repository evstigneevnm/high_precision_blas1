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

#ifndef __SCFD_DEFAULT_MONITOR_H__
#define __SCFD_DEFAULT_MONITOR_H__

#include <vector>
#include <utils/logged_obj_base.h>
#include <utils/log.h>
#include <numerical_algos/lin_solvers/detail/complex_calls_wrap.h>
#include <numerical_algos/detail/vectors_arr_wrap_static.h>

namespace numerical_algos
{
namespace lin_solvers 
{

template<class VectorOperations,class Log>
class default_monitor : public utils::logged_obj_base<Log>
{
public:
    typedef typename VectorOperations::scalar_type  scalar_type;
    typedef typename VectorOperations::vector_type  vector_type;
    typedef VectorOperations                        vector_operations_type;

private:
    typedef scalar_type                                         T;
    
    using T_real = typename detail::complex_base<T>::type;
    
    typedef utils::logged_obj_base<Log>                         logged_obj_t;
    typedef numerical_algos::detail::vectors_arr_wrap_static<VectorOperations,1> buf_arr_t;

    const vector_operations_type &vec_ops_;

    T_real           rel_tol_, abs_tol_;
    T_real           rel_tol_save_;
    int         max_iters_num_, min_iters_num_;
    int         max_iters_num_save_;
    bool        out_min_resid_norm_;
    bool        save_convergence_history_;
    bool        divide_out_norms_by_rel_base_;

    //followings are current convergence info
    int         iters_performed_;
    //is_valid_number is a flag, meaning whether current solution is a valid 
    //vector (without nans or infs)
    bool        is_valid_number_;
    T_real      rhs_norm_;
    T_real      resid_norm_;

    T_real      min_resid_norm_;
    buf_arr_t   buf_;
    vector_type &min_resid_norm_x_;

    std::vector<std::pair<int,T_real> >  convergence_history_;
public:
    default_monitor(const vector_operations_type &vec_ops, 
                    Log *log = NULL, int obj_log_lev = 0) : 
        logged_obj_t(log, obj_log_lev, "default_monitor: "),
        vec_ops_(vec_ops), buf_(&vec_ops),
        min_resid_norm_x_(buf_[0]) {}

    void                    init(T_real rel_tol, T_real abs_tol = T_real(0.f),
                                 int max_iters_num = 100, int min_iters_num = 0, 
                                 bool out_min_resid_norm = false,
                                 bool save_convergence_history = false,
                                 bool divide_out_norms_by_rel_base = false)
    {
        rel_tol_ = rel_tol; abs_tol_ = abs_tol;
        max_iters_num_ = max_iters_num; min_iters_num_ = min_iters_num;
        max_iters_num_save_ = max_iters_num;
        out_min_resid_norm_ = out_min_resid_norm;
        if (out_min_resid_norm_) buf_.init();
        save_convergence_history_ = save_convergence_history;
        divide_out_norms_by_rel_base_ = divide_out_norms_by_rel_base;
        rel_tol_save_ = rel_tol_;
    }
    void                    set_save_convergence_history(bool save_convergence_history)
    {
        save_convergence_history_ = save_convergence_history;
    }
    void                    set_divide_out_norms_by_rel_base(bool divide_out_norms_by_rel_base)
    {
        divide_out_norms_by_rel_base_ = divide_out_norms_by_rel_base;
    }
    //TODO init with config
    //TODO add separate function to control tolerances and behaviour
    void set_temp_tolerance(T rel_tol)
    {
        rel_tol_save_ = rel_tol_;
        rel_tol_ = rel_tol;   
    }
    void restore_tolerance()
    {
        rel_tol_ = rel_tol_save_;
    }
    void set_temp_max_iterations(int max_iter_local)
    {
        max_iters_num_save_ = max_iters_num_;
        max_iters_num_ = max_iter_local;
    }
    void restore_max_iterations()
    {
        max_iters_num_ = max_iters_num_save_;
    }


    void                    start(const vector_type& rhs)
    {
        rhs_norm_ = vec_ops_.norm(rhs);
        iters_performed_ = 0;
        if (out_min_resid_norm_) buf_.start_use_all();
        if (save_convergence_history_) convergence_history_.clear();
    }
    void                    stop()
    {
        if (out_min_resid_norm_) buf_.stop_use_all();
    }

    T_real                  rel_tol()const { return rel_tol_; }
    T_real                  abs_tol()const { return abs_tol_; }
    T_real                  rel_tol_base()const { return rhs_norm(); }
    T_real                  tol()const { return abs_tol() + rel_tol()*rel_tol_base(); }
    int                     max_iters_num()const { return max_iters_num_; } 
    int                     min_iters_num()const { return min_iters_num_; }
    bool                    out_min_resid_norm()const { return out_min_resid_norm_; }
    bool                    save_convergence_history()const { return save_convergence_history_; }
    bool                    divide_out_norms_by_rel_base()const { return divide_out_norms_by_rel_base_; }

    int                                     iters_performed()const { return iters_performed_; }
    bool                                    is_valid_number()const { return is_valid_number_; }
    T_real                                  rhs_norm()const { return rhs_norm_; }
    T_real                                  resid_norm()const { return resid_norm_; }
    T_real                                  resid_norm_out()const 
    { 
        if (!divide_out_norms_by_rel_base())
            return resid_norm(); 
        else
            return resid_norm()/rel_tol_base();
    }
    T_real tol_out()const 
    { 
        if (!divide_out_norms_by_rel_base())
            return tol(); 
        else
            return tol()/rel_tol_base();
    }
    const vector_type                       &min_resid_norm_x()const { return min_resid_norm_x_; }

    const std::vector<std::pair<int,T_real> >    &convergence_history()const { return convergence_history_; }

    default_monitor                         &operator++() 
    {  
        ++iters_performed_; 
        return *this;
    }
    default_monitor                         &operator+=(int n) 
    {  
        iters_performed_ += n; 
        return *this;
    }
    bool                                    converged()const
    {
        return is_valid_number() && (resid_norm() <= tol());
    }
    bool                                    check_finished(const vector_type& x, const vector_type& r)
    {
        logged_obj_t::info_f("iter = %d", iters_performed());

        is_valid_number_ = vec_ops_.check_is_valid_number(x);
        if (!is_valid_number_) {
            logged_obj_t::info_f("solution is not valid number");
            return true;
        }

        resid_norm_ = vec_ops_.norm(r);

        logged_obj_t::info_f("resid norm = %0.6e tol = %0.6e", resid_norm_out(), tol_out());
        if (save_convergence_history_) 
            convergence_history_.push_back( std::pair<int,T_real>(iters_performed(), resid_norm_out()) );

        if (out_min_resid_norm()) {
            if ((iters_performed() == 0)||(resid_norm() < min_resid_norm_)) {
                min_resid_norm_ = resid_norm();
                vec_ops_.assign(x, min_resid_norm_x_);
            }
        }
        
        return (converged() && iters_performed() >= min_iters_num()) || iters_performed() >= max_iters_num();
    }
};

}
}

#endif
