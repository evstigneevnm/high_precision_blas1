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
#ifndef __PARAMETERS_JSON_H__
#define __PARAMETERS_JSON_H__

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include<contrib/json/nlohmann/json.hpp>

namespace params
{

typedef enum 
{
    CPU = 0,
    GPU = 1
} __device_type;

std::string __print_bool(bool& val_)
{

    std::string ret = "false";
    if(val_)
    {
        ret = "true";   
    }
    return(ret);
}

std::string __print_enum(__device_type type_)
{
    std::string ret = "CPU";
    if(type_ == GPU)
    {
        ret = "GPU";
    }
    return(ret);
}

struct holder
{

    struct device
    {
        std::string type_string = "GPU";
        int pci_id = -1;
        unsigned int threads = 1;
        __device_type type;
        void plot_all()
        {
            if(type_string == "CPU")
                type = CPU;
            else if(type_string == "GPU")
                type = GPU;
            else
                throw std::logic_error("incorrect device selected: " + type_string + ". Only GPU or CPU are possible.");

            std::cout << "||  |==type: " << __print_enum(type) << std::endl;  
            if(type_string == "CPU")
            {
                std::cout << "||  |==threads: " << threads << std::endl;

            }
            else if(type_string == "GPU")
            {
                std::cout << "||  |==pci_id: " << pci_id << std::endl;
            }
        }

    };
    struct linear_solver
    {
        double rel_tol = 1.0e-6;
        bool use_preconditioned_residual = true;
        bool use_real_residual = true;
        bool save_convergence_history = true;
        bool divide_out_norms_by_rel_base = true;   
        bool verbose = false;
        struct bicgstabl_c
        {
            bool set = false;
            unsigned int max_iteration = 100;
            unsigned int basis_size = 3;
            void plot_all()
            {
                if(set)
                {
                    std::cout << "||  bicgstabl " << std::endl;
                    std::cout << "||  |    |==basis_size: " << basis_size << std::endl;
                    std::cout << "||  |    |==max_iteration: " << max_iteration << std::endl;                    
                }
            }
        };
        struct bicgstab_c
        {
            bool set = false;
            unsigned int max_iteration = 100;
            void plot_all()
            {
                if(set)
                {
                    std::cout << "||  bicgstab " << std::endl;
                    std::cout << "||  |    |==max_iteration: " << max_iteration << std::endl;                    
                }
            }
        };
        struct gmres_c
        {
            bool set = false;
            unsigned int max_iteration = 100;
            unsigned int restarts = 50;
            bool reorthogonalization = false;

            void plot_all()
            {
                if(set)
                {
                    std::cout << "||  gmres " << std::endl;
                    std::cout << "||  |    |==restarts: " << restarts << std::endl;
                    std::cout << "||  |    |==max_iteration: " << max_iteration << std::endl;                    
                    std::cout << "||  |    |==reorthogonalization: " << __print_bool(reorthogonalization) << std::endl;
                    if(max_iteration < restarts)
                    {
                        throw std::logic_error("gmres: maximum number of tterations cannot be smaller then the number of restarts (Krylov subspace basis size)");
                    }
                }
            }
        };        
        bicgstab_c bicgstab;
        bicgstabl_c bicgstabl;
        gmres_c gmres;
        void plot_all()
        {
            std::cout << "||  |==rel_tol: " << rel_tol << std::endl;
            std::cout << "||  |==use_preconditioned_residual: " << __print_bool(use_preconditioned_residual) << std::endl;
            std::cout << "||  |==use_real_residual: " << __print_bool(use_real_residual) << std::endl;
            std::cout << "||  |==save_convergence_history: " << __print_bool(save_convergence_history) << std::endl;
            std::cout << "||  |==divide_out_norms_by_rel_base: " << __print_bool(divide_out_norms_by_rel_base) << std::endl;
            std::cout << "||  |==verbose: " << __print_bool(verbose) << std::endl;
            bicgstabl.plot_all();
            bicgstab.plot_all();
            gmres.plot_all();
        }            
    };

    struct matrix
    {
        std::string file_name = "none";
        std::vector<std::string> prec_file_names = {"none"};
        void plot_all()
        {
            std::cout << "||  |==file_name: " << file_name << std::endl;
            if(prec_file_names[0] != "none")
            {
                for(auto &x: prec_file_names)
                {
                    std::cout << "||  |==preconditioner_file_name: " << x << std::endl;
                }
            }
        }
    };


    std::string convergence_file_name = "none";
    std::string condition_number_file_name = "none";
    std::string log_file_name = "none";
    device device_c;
    linear_solver linear_solver_c;
    matrix matrix_c;
    void plot_all()
    {
        std::cout << "CONFIGURATION:" << std::endl;
        std::cout << "||convergence_file_name: " << convergence_file_name << std::endl;
        std::cout << "||condition_number_file_name: " << condition_number_file_name << std::endl;
        std::cout << "||log_file_name: " << log_file_name << std::endl;
        std::cout << "||device:" << std::endl;
        device_c.plot_all();
        std::cout << "||linear solver:" << std::endl;
        linear_solver_c.plot_all();
        std::cout << "||matrix:" << std::endl;
        matrix_c.plot_all();
        std::cout << "ENDS" << std::endl;
    }    

};
using holder_t = holder;
 
void from_json(const nlohmann::json &j, holder_t::device &params_device_)
{
    int pci_id_ = -1;
    unsigned int threads_ = 1;
    try
    {
        pci_id_ = j.at("pci_id").get<int>();
        threads_ = j.at("threads").get<unsigned int>();
    }
    catch(const nlohmann::json::exception &exception)
    {
        std::cout << exception.what() << "...continuing." << std::endl;
    }

    params_device_ = holder_t::device
    {
        j.at("type").get<std::string>(),
        pci_id_,
        threads_
    };
}



void from_json(const nlohmann::json &j, holder_t::linear_solver::bicgstab_c& params_bicgstab_)
{
    params_bicgstab_ =  holder_t::linear_solver::bicgstab_c
    {
        true,
        j.at("maximum_iterations").get<unsigned int>()
    };

}
void from_json(const nlohmann::json &j, holder_t::linear_solver::bicgstabl_c& params_bicgstabl_)
{
    params_bicgstabl_ =  holder_t::linear_solver::bicgstabl_c
    {
        true,
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("basis_size").get<unsigned int>()
    };

}
void from_json(const nlohmann::json &j, holder_t::linear_solver::gmres_c& params_gmres_)
{
    params_gmres_ =  holder_t::linear_solver::gmres_c
    {
        true,
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("restarts").get<unsigned int>(),
        j.at("reorthogonalization").get<bool>()
    };

}
void from_json(const nlohmann::json &j, holder_t::linear_solver& params_lse_)
{
    
    bool none_set_error_ = true;
    holder_t::linear_solver::bicgstab_c bicgstab_loc;
    try
    {
        bicgstab_loc = j.at("bicgstab").get< holder_t::linear_solver::bicgstab_c >();
        none_set_error_ = false;
    }
    catch(const nlohmann::json::exception &exception)
    {
        std::cout << exception.what() << "...continuing." << std::endl;
    }
    holder_t::linear_solver::bicgstabl_c bicgstabl_loc;
    try
    {
        bicgstabl_loc = j.at("bicgstabl").get< holder_t::linear_solver::bicgstabl_c >();
        none_set_error_ = false;
    }
    catch(const nlohmann::json::exception &exception)
    {
        std::cout << exception.what() << "...continuing." << std::endl;
    }
    holder_t::linear_solver::gmres_c gmres_loc;
    try
    {
        gmres_loc = j.at("gmres").get< holder_t::linear_solver::gmres_c >();
        none_set_error_ = false;
    }
    catch(const nlohmann::json::exception &exception)
    {
        std::cout << exception.what() << "...continuing." << std::endl;
    }
    if(none_set_error_)
    {
        throw std::logic_error("not a singe linear solver is provided!");
    }

    params_lse_ = holder_t::linear_solver
    {
        j.at("relative_tolerance").get<double>(),
        j.at("use_preconditioned_residual").get<bool>(),
        j.at("use_real_residual").get<bool>(),
        j.at("save_convergence_history").get<bool>(),
        j.at("divide_norms_by_relative_base").get<bool>(),
        j.at("verbose").get<bool>(),
        bicgstab_loc,
        bicgstabl_loc,
        gmres_loc
    };
}

void from_json(const nlohmann::json &j, holder_t::matrix &params_mat_)
{
    
    std::string lin_op_f_n = "none";
    std::vector<std::string> prec_f_n = {"none"};
    try
    {
        lin_op_f_n = j.at("file_name").get<std::string>();
        prec_f_n = j.at("preconditioners").get< std::vector<std::string> >();
    }
    catch(const nlohmann::json::exception &exception)
    {
        std::cout << exception.what() << "...continuing." << std::endl;
    }

    params_mat_ = holder_t::matrix
    {
        lin_op_f_n,
        prec_f_n      
    };
}

void from_json(const nlohmann::json &j, holder_t &params_)
{
    params_ = holder_t
    {
        j.at("convergence_file_name").get<std::string>(),
        j.at("condition_number_file_name").get<std::string>(),
        j.at("log_file_name").get<std::string>(),
        j.at("device").get< holder_t::device >(),
        j.at("linear_solver").get< holder_t::linear_solver >(),    
        j.at("matrix").get< holder_t::matrix >()
    };
}

nlohmann::json read_json(const std::string &project_file_name_)
{
    try
    {
        std::ifstream f(project_file_name_);
        if (f)
        {
            nlohmann::json j;
            f >> j;
            return j;
        }
        else
        {
            throw std::runtime_error(std::string("Failed to open file ") + project_file_name_ + " for reading");
        }
    }
    catch (const nlohmann::json::exception &exception)
    {
        std::throw_with_nested(std::runtime_error{"json path: " + project_file_name_ + "\n" + exception.what()});
    }
}


holder read_holder_json(const std::string &project_file_name_)
{
    holder holder_str;
    try
    {
        holder_str = read_json(project_file_name_).get< holder >();
    }
    catch(const std::exception& e)
    {
        std::throw_with_nested(std::runtime_error{std::string("failed to read json file: ") + std::string("\n") + e.what()});
    }
    return holder_str;
}


}
#endif