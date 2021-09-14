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
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <linear_operator.hpp>
#include <utils/log.h>
#include <common/cpu_vector_operations_child.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstab.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/gmres.h>

typedef TYPE T;

void write_convergency(const std::string &fn, const std::vector<std::pair<int,T> > &conv, T tol)
{
    std::ofstream f(fn.c_str(), std::ofstream::out);
    if (!f) throw std::runtime_error("write_convergency: error while opening file " + fn);

    for (int i = 0;i < conv.size();++i) {
        if (!(f << conv[i].first << " " << conv[i].second << " " << tol << std::endl)) 
            throw std::runtime_error("write_convergency: error while writing to file " + fn);
    }
}

void write_condition_number(const std::string &file_name, const std::vector<double> &data)
{
    std::ofstream f(file_name, std::ofstream::out);
    if(!f)
    {
        throw std::runtime_error("write_condition_number: error while opening file " + file_name);
    }
    for(auto &x: data)
    {
        if (!(f << x << std::endl)) 
        {
            throw std::runtime_error("write_condition_number: error while writing to file " + file_name);
        }
    }
}

int main(int argc, char const *argv[])
{

    if (argc < 14) {
        std::cout << "USAGE: " << std::string(argv[0]) << " <mesh_sz> <a> <re> <max_iters> <rel_tol> <use_precond_resid> <use_real_resid> <basis_sz> <lin_solver_type> <result_fn> <convergency_fn> <use_high_precision_dot_prod (1/0)> <number_of_threads> "  << std::endl;
        std::cout << "EXAMPLE: " << std::string(argv[0]) << " 100 1. 100. 100 1e-7 1 0 2 bicgstab test_out.dat conv_out.dat 0 1"  << std::endl;
        return 0;
    }
    std::string             res_fn(argv[10]), conv_fn(argv[11]);
    int                     sz = atoi(argv[1]);
    T                       a = atof(argv[2]),
                            re = atoi(argv[3]);
    int                     max_iters = atoi(argv[4]);
    T                       rel_tol = atof(argv[5]);
    bool                    use_precond_resid = atoi(argv[6]);
    bool                    use_real_resid = atoi(argv[7]);
    int                     basis_sz = atoi(argv[8]);
    std::string lin_solver_type(argv[9]);
    int                     use_high_precision_dot_prod = atoi(argv[12]);
    int                     use_threaded_dot_prod = atoi(argv[13]);

    using vec_ops_t = cpu_vector_operations_child<T>;
    using T_vec = typename vec_ops_t::vector_type;

    using sys_op_t = SystemOperator<vec_ops_t>;
    using precond_t = prec_operator<vec_ops_t, sys_op_t>;

    using log_t = utils::log_std;
    using mon_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t, log_t>;
    using bicgstab_t = numerical_algos::lin_solvers::bicgstab<sys_op_t,precond_t,vec_ops_t,mon_t,log_t>;
    using bicgstabl_t = numerical_algos::lin_solvers::bicgstabl<sys_op_t,precond_t,vec_ops_t,mon_t,log_t>;
    using gmres_t = numerical_algos::lin_solvers::gmres<sys_op_t,precond_t,vec_ops_t,mon_t,log_t>;


    vec_ops_t vec_ops(sz, use_high_precision_dot_prod, use_threaded_dot_prod);
    sys_op_t sys_op(&vec_ops, a, re);
    precond_t prec;
    log_t log;    
    bicgstabl_t bicgstabl(&vec_ops, &log);
    gmres_t gmres(&vec_ops, &log);
    bicgstab_t bicgstab(&vec_ops, &log);
    mon_t *mon;
    
    if(lin_solver_type == "bicgstabl")
    {
        mon = &bicgstabl.monitor();
        std::cout << "using bcgstabl(" << basis_sz << ")" << std::endl;
    }
    else if(lin_solver_type == "gmres")
    {
        mon = &gmres.monitor();
        gmres.set_reorthogonalization(false);
        std::cout << "using gmres with " << basis_sz << " restarts" << std::endl;    
    }
    else if(lin_solver_type == "bicgstab")
    {
        mon = &bicgstab.monitor();
        std::cout << "using bcgstab" << std::endl;    
    }
    else
    {
        throw(std::runtime_error(std::string("selected linear solver is not supported!")));
    }

    T_vec x, rhs, r;
    vec_ops.init_vector(x); vec_ops.start_use_vector(x);
    vec_ops.init_vector(rhs); vec_ops.start_use_vector(rhs);
    vec_ops.init_vector(r); vec_ops.start_use_vector(r);
    vec_ops.assign_scalar(T(1.0), rhs);
    vec_ops.assign_scalar(T(0.0), x);
    vec_ops.assign_scalar(T(0.0), r);

    mon->init(rel_tol, T(0.0), max_iters);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    bicgstabl.set_preconditioner(&prec);    
    gmres.set_preconditioner(&prec);
    bicgstab.set_preconditioner(&prec);
    
    bicgstabl.set_resid_recalc_freq(int(use_real_resid));
    bicgstabl.set_use_precond_resid(use_precond_resid);    
    bicgstabl.set_basis_size(basis_sz);
    
    gmres.set_use_precond_resid(use_precond_resid);    
    gmres.set_restarts(basis_sz);
    gmres.set_reorthogonalization(bool(use_real_resid));
    
    bicgstab.set_use_precond_resid(use_precond_resid);

    bool res_flag_ = false;

    auto start = std::chrono::steady_clock::now();
    if(lin_solver_type == "bicgstabl")
        res_flag_ = bicgstabl.solve(sys_op, rhs, x);
    else if(lin_solver_type == "gmres") 
        res_flag_ = gmres.solve(sys_op, rhs, x);
    else if(lin_solver_type == "bicgstab") 
        res_flag_ = bicgstab.solve(sys_op, rhs, x);
    auto finish = std::chrono::steady_clock::now();
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(finish - start).count();
    std::cout << "execution wall time = " << elapsed_seconds << "sec." << std::endl;    

    sys_op.apply(x, r);
    vec_ops.add_mul(T(1.0), rhs, -T(1.0), r);
    std::cout << std::scientific << "actual residual norm = " << vec_ops.norm(r) << std::endl;


    write_condition_number("condition_number_"+std::to_string(use_high_precision_dot_prod)+".dat", vec_ops.get_condition_numbers() );

    int iters_performed_ = mon->iters_performed();

    if (res_flag_) 
        log.info("lin_solver returned success result");
    else
        log.info("lin_solver returned fail result");

    if (res_fn != "none") {
        std::ofstream    out_f(res_fn.c_str());
        for (int i = 0;i < sz;++i) {
            out_f << (i + T(0.5f))/sz << " " << x[i] << std::endl;
        }
        out_f.close();
    }
    if (conv_fn != "none") 
        write_convergency(conv_fn, mon->convergence_history(), mon->tol_out());

    vec_ops.stop_use_vector(x); vec_ops.free_vector(x);
    vec_ops.stop_use_vector(rhs); vec_ops.free_vector(rhs);    
    vec_ops.stop_use_vector(r); vec_ops.free_vector(r);

    return 0;
}