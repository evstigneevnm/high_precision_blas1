#include <thrust/complex.h>
#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_vector_operations_child.h>
#include <common/csr/gpu_matrix.h>
#include <common/csr/linear_operator.h>
#include <common/csr/preconditioner.h>
#include <common/csr/gpu_ilu0_preconditioner.h>
#include <common/csr/matrix_market_reader.h>
#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstab.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/gmres.h>
#include <parameters/parameters_json.h>
#include <generate_vector_pair_complex.hpp>
#include <high_prec/dot_product_gmp.hpp>


std::string insert_string(const std::string original, const std::string new_string, const std::string delimiter = ".")
{
    size_t pos = 0;
    std::string token;
    std::string s = original;
    while ((pos = s.find(delimiter)) != std::string::npos)
    {
        token = s.substr(0, pos);
        // std::cout << "token: " <<token << std::endl;
        s.erase(0, pos + delimiter.length());
        token = token+new_string+delimiter+s;
        break;
    } 
    return(token);
}

template<class T>
void write_convergency(const std::string &fn, const std::vector<std::pair<int,T> > &conv, T tol_)
{
    std::ofstream f(fn.c_str(), std::ofstream::out);
    if (!f) throw std::runtime_error("write_convergency: error while opening file " + fn);

    for (int i = 0;i < conv.size();++i) {
        if (!(f << conv[i].first << " " << conv[i].second << " " << tol_ << std::endl)) 
            throw std::runtime_error("write_convergency: error while writing to file " + fn);
    }
}

template<class T>
struct print_basic_type
{
};
template<>
struct print_basic_type<float>
{
    std::string name = "float";
};
template<>
struct print_basic_type<double>
{
    std::string name = "double";
};
template<>
struct print_basic_type<thrust::complex<float> >
{
    std::string name = "float";
};
template<>
struct print_basic_type<thrust::complex<double> >
{
    std::string name = "double";
};

int main(int argc, char const *argv[])
{
    using T_base = TYPE;
    using TC = thrust::complex<T_base>;
    using T  = TC;
    
    using vec_ops_t = gpu_vector_operations<T>;
    using vec_ops_double_t = gpu_vector_operations<double>;
    using vec_ops_real_t = gpu_vector_operations<T_base>;
    using dot_exact_t = dot_product_gmp<double, double*>;
    using generate_vector_pair_t = generate_vector_pair_complex<vec_ops_t,vec_ops_real_t, vec_ops_double_t, dot_exact_t>;

    using vec_ops_cn_t = gpu_vector_operations_child<T, generate_vector_pair_t>;

    using T_vec = vec_ops_t::vector_type;
    using T_mat_t = csr::gpu_matrix<vec_ops_t, cublas_wrap>;
    using mm_reader_t = csr::matrix_market_reader<T>;
    using lin_op_t = csr::linear_operator<vec_ops_t, T_mat_t>;
    using prec_t = csr::preconditioner<vec_ops_t, lin_op_t, T_mat_t>;
    using ilu0_prec_t = csr::gpu_ilu0_preconditioner<vec_ops_t, lin_op_t, T_mat_t>;
    using params_t = params::holder;

    using log_t = utils::log_std;
    using mon_t = numerical_algos::lin_solvers::default_monitor<vec_ops_cn_t, log_t>;
    using bicgstab_t = numerical_algos::lin_solvers::bicgstab<lin_op_t,prec_t,vec_ops_cn_t,mon_t,log_t>;
    using bicgstabl_t = numerical_algos::lin_solvers::bicgstabl<lin_op_t,prec_t,vec_ops_cn_t,mon_t,log_t>;
    using gmres_t = numerical_algos::lin_solvers::gmres<lin_op_t,prec_t,vec_ops_cn_t,mon_t,log_t, 300>;



    if(argc!=2)
    {
        std::cout << "usage: " << argv[0] << " parameters_file.json" << std::endl;
        std::cout << "where parameters_file.json is the configuration file." << std::endl;
        return(0);
    }
    print_basic_type<T> pbt;
    std::string basic_type_name = pbt.name;

    std::string path_to_config_file_(argv[1]);
    params_t parameters = params::read_holder_json(path_to_config_file_);
    parameters.plot_all();
    
    if(parameters.device_c.type == params::GPU)
    {
        init_cuda(parameters.device_c.pci_id);
    }
    cublas_wrap cublas(true);
    T_mat_t mat(&cublas);
    T_mat_t matL(&cublas);
    T_mat_t matU(&cublas);
    mm_reader_t reader(true);
    bool external_precond = false;
    if(parameters.matrix_c.prec_file_names[0] != "none")
    {
        external_precond = true;
        reader.read_file(parameters.matrix_c.prec_file_names[0]);
        reader.set_csr_matrix<T_mat_t>(&matL);
        reader.read_file(parameters.matrix_c.prec_file_names[1]);
        reader.set_csr_matrix<T_mat_t>(&matU);
    }
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
    vec_ops_double_t vec_ops_double(sz, &cublas);
    vec_ops_real_t vec_ops_real(sz, &cublas);
    dot_exact_t dot_exact;
    generate_vector_pair_t generate_vector_pair(&vec_ops, &vec_ops_real, &vec_ops_double, &dot_exact);    
    vec_ops_cn_t vec_ops_cn(sz, &cublas);
    vec_ops_cn.set_condition_number_helper(&generate_vector_pair);


    lin_op_t lin_op(&vec_ops, &mat);
    prec_t prec(&vec_ops);
    ilu0_prec_t ilu0_prec(&vec_ops);

    log_t log; 
    log.set_verbosity(3);   
    bicgstabl_t bicgstabl(&vec_ops_cn, &log);
    gmres_t gmres(&vec_ops_cn, &log);
    bicgstab_t bicgstab(&vec_ops_cn, &log);
    mon_t *mon_gmres, *mon_bicgstabl, *mon_bicgstab;    

    std::vector<std::string> linear_solver_names;
    
    if(parameters.linear_solver_c.verbose)
    {
        linear_solver_names.reserve(3);
    }

    if(parameters.linear_solver_c.bicgstabl.set)
    {
        mon_bicgstabl = &bicgstabl.monitor();
        bicgstabl.set_preconditioner(&prec);
        bicgstabl.set_resid_recalc_freq(int(parameters.linear_solver_c.use_real_residual));
        bicgstabl.set_use_precond_resid(parameters.linear_solver_c.use_preconditioned_residual);
        bicgstabl.set_basis_size(parameters.linear_solver_c.bicgstabl.basis_size);        

       
        mon_bicgstabl->init(T_base(parameters.linear_solver_c.rel_tol), 0, parameters.linear_solver_c.bicgstabl.max_iteration);
        mon_bicgstabl->set_save_convergence_history(parameters.linear_solver_c.save_convergence_history);
        mon_bicgstabl->set_divide_out_norms_by_rel_base(parameters.linear_solver_c.divide_out_norms_by_rel_base); 

        std::cout << "using bcgstabl(" << parameters.linear_solver_c.bicgstabl.basis_size << ")." << std::endl;
        linear_solver_names.push_back("bicgstabl");
    }
    if(parameters.linear_solver_c.gmres.set)
    {
        mon_gmres = &gmres.monitor();
        gmres.set_preconditioner(&prec);
        gmres.set_reorthogonalization(parameters.linear_solver_c.gmres.reorthogonalization);
        gmres.set_use_precond_resid(parameters.linear_solver_c.use_preconditioned_residual);    
        gmres.set_restarts(parameters.linear_solver_c.gmres.restarts);
        gmres.set_resid_recalc_freq(int(parameters.linear_solver_c.use_real_residual));
        
        mon_gmres->init(T_base(parameters.linear_solver_c.rel_tol), 0, parameters.linear_solver_c.gmres.max_iteration);
        mon_gmres->set_save_convergence_history(parameters.linear_solver_c.save_convergence_history);
        mon_gmres->set_divide_out_norms_by_rel_base(parameters.linear_solver_c.divide_out_norms_by_rel_base); 

        std::cout << "using gmres with " << parameters.linear_solver_c.gmres.restarts << " restarts." << std::endl;    
        linear_solver_names.push_back("gmres");
    }
    if(parameters.linear_solver_c.bicgstab.set)
    {
        mon_bicgstab = &bicgstab.monitor();
        bicgstab.set_preconditioner(&prec);
        bicgstab.set_use_precond_resid(parameters.linear_solver_c.use_preconditioned_residual);
        
        mon_bicgstab->init(T_base(parameters.linear_solver_c.rel_tol), 0, parameters.linear_solver_c.bicgstab.max_iteration);
        mon_bicgstab->set_save_convergence_history(parameters.linear_solver_c.save_convergence_history);
        mon_bicgstab->set_divide_out_norms_by_rel_base(parameters.linear_solver_c.divide_out_norms_by_rel_base);         
        std::cout << "using bcgstab." << std::endl; 
        linear_solver_names.push_back("bicgstab");   
    }
    linear_solver_names.shrink_to_fit();


    T_vec x; T_vec b; T_vec r;
    vec_ops.init_vector(x); vec_ops.start_use_vector(x);
    vec_ops.init_vector(b); vec_ops.start_use_vector(b);
    vec_ops.init_vector(r); vec_ops.start_use_vector(r);
   
    if(parameters.matrix_c.prec_file_names[0] != "none")
    {
        prec.set_matrix(&matL, &matU);
    }
    else
    {
        ilu0_prec.set_matrix(mat);
    }

    std::string log_f_n_ = parameters.log_file_name;
    if(log_f_n_ != "none")
    {
        log_f_n_ = insert_string(log_f_n_, "_" + basic_type_name);
    }

    if(log_f_n_ != "none")
    {
        std::ofstream f(log_f_n_.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("log_file_name: error while opening file " + parameters.log_file_name);

        f << "basic type: " << basic_type_name << std::endl;
        f << "matrices: " << parameters.matrix_c.file_name;
        if(parameters.matrix_c.prec_file_names[0] != "none")
        {
            for(auto &_name: parameters.matrix_c.prec_file_names)
            {
                f << ", " <<  _name;
            }
        }
        f << std::endl << "=========================" << std::endl;
        f.close();
    }

    for (std::string &ls_name_: linear_solver_names)
    {
        mon_t *mon_;
        for(bool h_prec_: {false, true} )
        {
            vec_ops_cn.reset_condition_number_storage();
            std::cout << "solve " << ls_name_ << " ";
            std::string prec_ = "standard precision";
            if(h_prec_)
            {
                vec_ops.use_high_precision();
                std::cout << "using high precision." << std::endl;
                prec_ = "high precision";
            }
            else
            {
                vec_ops.use_standard_precision();
                std::cout << "using standard precision." << std::endl;
            }
            vec_ops.assign_scalar(T(1.0), b);
            vec_ops.assign_scalar(T(0.0), x);
            vec_ops.assign_scalar(T(0.0), r); 
            bool res_flag_ = false;
            int iters_performed_ = 0;
            auto start = std::chrono::steady_clock::now();
            if(ls_name_ == "bicgstabl")
            {
                res_flag_ = bicgstabl.solve(lin_op, b, x);
                iters_performed_ = mon_bicgstabl->iters_performed();
                mon_ = mon_bicgstabl;
            }
            else if(ls_name_ == "gmres") 
            {
                res_flag_ = gmres.solve(lin_op, b, x);
                iters_performed_ = mon_gmres->iters_performed();
                mon_ = mon_gmres;
            }
            else if(ls_name_ == "bicgstab") 
            {
                res_flag_ = bicgstab.solve(lin_op, b, x);
                iters_performed_ = mon_bicgstab->iters_performed();
                mon_ = mon_bicgstab;
            }
            auto finish = std::chrono::steady_clock::now();
            double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(finish - start).count();
            std::cout.precision(3);
            std::cout << std::defaultfloat << "execution wall time = " << elapsed_seconds << "sec." << std::endl;    

            lin_op.apply(x, r);
            vec_ops.add_mul(T(1.0), b, -T(1.0), r);
            
            std::cout.precision(6);
            std::cout << std::scientific << "actual residual norm = " << vec_ops.norm(r) << std::endl;
            if (res_flag_) 
                log.info_f("%s returned success result", ls_name_.c_str() );
            else
                log.info_f("%s returned fail result", ls_name_.c_str() );

            
            if(parameters.condition_number_file_name != "none")
            {
                std::string _pr_file_name = "_"+ basic_type_name + "_" + ls_name_ + "_s_p";
                if(h_prec_)
                {
                    _pr_file_name = "_"+ basic_type_name + "_" + ls_name_ + "_h_p";
                }
                std::string cond_file_name = insert_string(parameters.condition_number_file_name, _pr_file_name);
                vec_ops_cn.write_condition_number_file(cond_file_name);    
            }
            

            if (parameters.convergence_file_name != "none")
            { 
                std::string _pr_file_name = "_"+ basic_type_name + "_" + ls_name_ + "_s_p";
                if(h_prec_)
                {
                    _pr_file_name = "_"+ basic_type_name + "_" + ls_name_ + "_h_p";
                }

                std::string conv_file_name = insert_string(parameters.convergence_file_name, _pr_file_name);
                
                write_convergency<T_base>(conv_file_name, mon_->convergence_history(), mon_->tol_out() );
            
            }

            if(log_f_n_ != "none")
            {
                std::ofstream f(log_f_n_.c_str(), std::ios_base::app);
                if (!f) throw std::runtime_error("log_file_name: error while appending file " + parameters.log_file_name);
                f << "solver: " << ls_name_ << " with " << prec_;
                if(res_flag_)
                    f << " converged";
                else
                    f << " not converged";
                f << std::endl;
                f << "wall time: " << elapsed_seconds << "sec." << std::endl;
                f << "norm: " << vec_ops.norm(r) << std::endl;
                f << "=========================" << std::endl;
                f.close();
            }
        }
    }


    vec_ops.stop_use_vector(r); vec_ops.free_vector(r);
    vec_ops.stop_use_vector(b); vec_ops.free_vector(b);
    vec_ops.stop_use_vector(x); vec_ops.free_vector(x);    
    return 0;
}