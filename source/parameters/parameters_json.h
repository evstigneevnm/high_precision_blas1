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
        std::string name = "bicgstab";
        unsigned int max_iteration = 100;
        double rel_tol = 1.0e-6;
        bool use_preconditioned_residual = true;
        bool use_real_residual = true;
        unsigned int basis_size = 3;
        bool save_convergence_history;
        bool divide_out_norms_by_rel_base;    
      
        void plot_all()
        {
            std::cout << "||  |==name: " << name << std::endl;
            std::cout << "||  |==max_iteration: " << max_iteration << std::endl;
            std::cout << "||  |==rel_tol: " << rel_tol << std::endl;
            std::cout << "||  |==use_preconditioned_residual: " << __print_bool(use_preconditioned_residual) << std::endl;
            std::cout << "||  |==use_real_residual: " << __print_bool(use_real_residual) << std::endl;
            std::cout << "||  |==basis_size: " << basis_size << std::endl;                
            std::cout << "||  |==save_convergence_history: " << __print_bool(save_convergence_history) << std::endl;
            std::cout << "||  |==divide_out_norms_by_rel_base: " << __print_bool(divide_out_norms_by_rel_base) << std::endl;
        }            
    };

    struct matrix
    {
        std::string file_name = "none";
        void plot_all()
        {
            std::cout << "||  |==file_name: " << file_name << std::endl;
        }
    };


    bool use_high_precision = false;
    std::string convergence_file_name = "none";
    std::string condition_number_file_name = "none";

    device device_c;
    linear_solver linear_solver_c;
    matrix matrix_c;
    void plot_all()
    {
        std::cout << "CONFIGURATION:" << std::endl;
        std::cout << "||use_high_precision: " << __print_bool(use_high_precision) << std::endl;
        std::cout << "||convergence_file_name: " << convergence_file_name << std::endl;
        std::cout << "||condition_number_file_name: " << condition_number_file_name << std::endl;
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
    params_device_ = holder_t::device
    {
        j.at("type").get<std::string>(),
        j.at("pci_id").get<int>(),
        j.at("threads").get<unsigned int>()
    };
}

void from_json(const nlohmann::json &j, holder_t::linear_solver& params_lse_)
{
    params_lse_ = holder_t::linear_solver
    {
        j.at("name").get<std::string>(),
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("relative_tolerance").get<double>(),
        j.at("use_preconditioned_residual").get<bool>(),
        j.at("use_real_residual").get<bool>(),
        j.at("basis_size").get<unsigned int>(),
        j.at("save_convergence_history").get<bool>(),
        j.at("divide_norms_by_relative_base").get<bool>()
    };
}

void from_json(const nlohmann::json &j, holder_t::matrix &params_mat_)
{
    params_mat_ = holder_t::matrix
    {
        j.at("file_name").get<std::string>()        
    };
}

void from_json(const nlohmann::json &j, holder_t &params_)
{
    params_ = holder_t
    {
        j.at("use_high_precision").get<bool>(),
        j.at("convergence_file_name").get<std::string>(),
        j.at("condition_number_file_name").get<std::string>(),
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