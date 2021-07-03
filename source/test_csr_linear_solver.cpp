#include <thrust/complex.h>
#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <common/csr/gpu_matrix.h>
#include <common/csr/linear_operator.h>
#include <common/csr/preconditioner.h>
#include <common/csr/matrix_market_reader.h>
#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstab.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/gmres.h>
#include <parameters/parameters_json.h>


template<class T>
void write_convergency(const std::string &fn, const std::vector<std::pair<int,T> > &conv, T tol)
{
    std::ofstream f(fn.c_str(), std::ofstream::out);
    if (!f) throw std::runtime_error("write_convergency: error while opening file " + fn);

    for (int i = 0;i < conv.size();++i) {
        if (!(f << conv[i].first << " " << conv[i].second << " " << tol << std::endl)) 
            throw std::runtime_error("write_convergency: error while writing to file " + fn);
    }
}


int main(int argc, char const *argv[])
{
    using T = TYPE;
    using TC = thrust::complex<T>;
    using vec_ops_t = gpu_vector_operations<T>;
    using T_vec = vec_ops_t::vector_type;
    using T_mat_t = csr::gpu_matrix<vec_ops_t, cublas_wrap>;
    using mm_reader_t = csr::matrix_market_reader<T>;
    using lin_op_t = csr::linear_operator<vec_ops_t, T_mat_t>;
    using prec_t = csr::preconditioner<vec_ops_t, lin_op_t, T_mat_t>;
    using params_t = params::holder;

    using log_t = utils::log_std;
    using mon_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t, log_t>;
    using bicgstab_t = numerical_algos::lin_solvers::bicgstab<lin_op_t,prec_t,vec_ops_t,mon_t,log_t>;
    using bicgstabl_t = numerical_algos::lin_solvers::bicgstabl<lin_op_t,prec_t,vec_ops_t,mon_t,log_t>;
    using gmres_t = numerical_algos::lin_solvers::gmres<lin_op_t,prec_t,vec_ops_t,mon_t,log_t>;


    if(argc!=2)
    {
        std::cout << "usage: " << argv[0] << " parameters_file.json" << std::endl;
        std::cout << "where parameters_file.json is the configuration file." << std::endl;
        return(0);
    }
    std::string path_to_config_file_(argv[1]);
    params_t parameters = params::read_holder_json(path_to_config_file_);
    parameters.plot_all();
    
    if(parameters.device_c.type == params::GPU)
    {
        init_cuda(parameters.device_c.pci_id);
    }
    cublas_wrap cublas(true);
    T_mat_t mat(&cublas);
    mm_reader_t reader(true);
    reader.read_file(parameters.matrix_c.file_name);
    reader.set_csr_matrix<T_mat_t>(&mat);
    auto dims = mat.get_dim();
    int size_y = dims.rows;
    int sz = dims.columns;
    if(size_y!=sz)
    {
        throw std::runtime_error("supplied matrix " + parameters.matrix_c.file_name + " is not square.");
    }

    vec_ops_t vec_ops(sz, &cublas);
    if(parameters.use_high_precision)
    {
        vec_ops.use_high_precision();
    }
    else
    {
        vec_ops.use_standard_precision();
    }

    lin_op_t lin_op(&vec_ops, &mat);
    prec_t prec(&vec_ops);

    log_t log;    
    bicgstabl_t bicgstabl(&vec_ops, &log);
    gmres_t gmres(&vec_ops, &log);
    bicgstab_t bicgstab(&vec_ops, &log);
    mon_t *mon;    

    if(parameters.linear_solver_c.name == "bicgstabl")
    {
        mon = &bicgstabl.monitor();
        bicgstabl.set_preconditioner(&prec);
        bicgstabl.set_resid_recalc_freq(int(parameters.linear_solver_c.use_real_residual));
        bicgstabl.set_use_precond_resid(parameters.linear_solver_c.use_preconditioned_residual);
        bicgstabl.set_basis_size(parameters.linear_solver_c.basis_size);        
        std::cout << "using bcgstabl(" << parameters.linear_solver_c.basis_size << ")" << std::endl;
    }
    else if(parameters.linear_solver_c.name == "gmres")
    {
        mon = &gmres.monitor();
        gmres.set_preconditioner(&prec);
        gmres.set_reorthogonalization(false);
        gmres.set_use_precond_resid(parameters.linear_solver_c.use_preconditioned_residual);    
        gmres.set_restarts(parameters.linear_solver_c.basis_size);
        gmres.set_resid_recalc_freq(int(parameters.linear_solver_c.use_real_residual));

        std::cout << "using gmres with " << parameters.linear_solver_c.basis_size << " restarts" << std::endl;    
    }
    else if(parameters.linear_solver_c.name == "bicgstab")
    {
        mon = &bicgstab.monitor();
        bicgstab.set_preconditioner(&prec);
        bicgstab.set_use_precond_resid(parameters.linear_solver_c.use_preconditioned_residual);
        std::cout << "using bcgstab" << std::endl;    
    }
    else
    {
        throw(std::runtime_error(std::string("selected linear solver is not supported!")));
    }

    mon->init(parameters.linear_solver_c.rel_tol, T(0.0), parameters.linear_solver_c.max_iteration);
    mon->set_save_convergence_history(parameters.linear_solver_c.save_convergence_history);
    mon->set_divide_out_norms_by_rel_base(parameters.linear_solver_c.divide_out_norms_by_rel_base);    

    T_vec x; T_vec b; T_vec r;
    vec_ops.init_vector(x); vec_ops.start_use_vector(x);
    vec_ops.init_vector(b); vec_ops.start_use_vector(b);
    vec_ops.init_vector(r); vec_ops.start_use_vector(r);
    vec_ops.assign_scalar(T(1.0), b);
    vec_ops.assign_scalar(T(0.0), x);
    vec_ops.assign_scalar(T(0.0), r);    

    bool res_flag_ = false;

    auto start = std::chrono::steady_clock::now();
    if(parameters.linear_solver_c.name == "bicgstabl")
        res_flag_ = bicgstabl.solve(lin_op, b, x);
    else if(parameters.linear_solver_c.name == "gmres") 
        res_flag_ = gmres.solve(lin_op, b, x);
    else if(parameters.linear_solver_c.name == "bicgstab") 
        res_flag_ = bicgstab.solve(lin_op, b, x);
    auto finish = std::chrono::steady_clock::now();
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(finish - start).count();
    std::cout << "execution wall time = " << elapsed_seconds << "sec." << std::endl;    

    lin_op.apply(x, r);
    vec_ops.add_mul(T(1.0), b, -T(1.0), r);
    std::cout << std::scientific << "actual residual norm = " << vec_ops.norm(r) << std::endl;
    int iters_performed_ = mon->iters_performed();

    if (res_flag_) 
        log.info("lin_solver returned success result");
    else
        log.info("lin_solver returned fail result");
    if (parameters.convergence_file_name != "none")
    { 
        write_convergency(parameters.convergence_file_name, mon->convergence_history(), mon->tol_out());
    }

    vec_ops.stop_use_vector(r); vec_ops.free_vector(r);
    vec_ops.stop_use_vector(b); vec_ops.free_vector(b);
    vec_ops.stop_use_vector(x); vec_ops.free_vector(x);    
    return 0;
}