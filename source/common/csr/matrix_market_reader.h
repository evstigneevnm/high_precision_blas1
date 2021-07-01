#ifndef __CSR__MATRIX_MARKET_READER_H__
#define __CSR__MATRIX_MARKET_READER_H__

/**
 * @brief      This class is a matrix market reader. For complex matrices the std::complex or thrust::complex is used. The idea is taken from cusp. Use matrix_market_reader_set_val.h to set up custom complex types
 *  
 * @tparam     T     { base type }
 */

#include<complex>
#include<vector>
#include<string>
#include<fstream>
#include<sstream>
#include<iostream>
#include<stdexcept>
#include<algorithm>
#include<chrono>
#include <type_traits>
#include<common/csr/matrix_market_reader_set_val.h>

namespace csr
{

template<class T, class Ord = size_t>
class matrix_market_reader
{
private:
    using T_real = typename complex_base_type<T>::real;


    struct csr_matrix
    {
        Ord row_p = 0;
        Ord col_i = 0;
        Ord nnz = 0;
        Ord num_cols = 0;
        Ord num_rows = 0;
        T* data = nullptr;
        Ord* row = nullptr;
        Ord* col = nullptr;
        ~csr_matrix()
        {
            if(data!= nullptr)
            {
                free(data);
            }
            if(row!= nullptr)
            {
                free(row);
            } 
            if(col!= nullptr)
            {
                free(col);
            } 
                       
        }
        void allocate()
        {
            
            row_p = num_rows+1;
            col_i = nnz;

            if(num_rows > 0)
            {
                row = (Ord*)malloc(sizeof(Ord)*row_p);
                if(row == NULL)
                {
                    throw std::runtime_error("csr::matrix_market_reader::csr_matrix: failed to malloc row array.");
                }
                for(Ord j = 0; j < row_p; j++)
                {
                    row[j] = 0;
                }
            }
            if(col_i > 0)
            {
                col = (Ord*)malloc(sizeof(Ord)*col_i);
                if(col == NULL)
                {
                    throw std::runtime_error("csr::matrix_market_reader::csr_matrix: failed to malloc col array.");
                }
                for(Ord j = 0; j < nnz; j++)
                {
                    col[j] = 0;
                }                
            }
            if(nnz > 0)
            {
                data = (T*)malloc(sizeof(T)*nnz);
                if(data == NULL)
                {
                    throw std::runtime_error("csr::matrix_market_reader::csr_matrix: failed to malloc data array.");
                }
            } 
        }

    };
    csr_matrix csr_mat;

    struct coo_matrix_node
    {
        Ord row;
        Ord col;
        T val;
    };
    using node = coo_matrix_node;
    struct coo_matrix
    {
        Ord nnz = 0;
        Ord num_cols = 0;
        Ord num_rows = 0;
        std::vector<coo_matrix_node> data;
        void allocate()
        {
            data.clear(); data.reserve(nnz);
        }
    };    

public:

    matrix_market_reader(bool verbose_ = false):
    verbose(verbose_)
    {
    
    }
    ~matrix_market_reader()
    {
    
    }
    template<class Matrix>
    void read_file_2_csr_marix(const std::string& file_name_, Matrix* mat_)
    {
        
        read_file(file_name_);
        mat_->init(csr_mat.nnz, csr_mat.col_i, csr_mat.row_p);
        mat_->set_dim(csr_mat.num_cols, csr_mat.num_rows);
        mat_->set( csr_mat.data, csr_mat.col, csr_mat.row );
    }
    
    template<class Matrix>
    void set_csr_matrix(Matrix* mat_)
    {
        if(matrix_file_read)
        {
            mat_->init(csr_mat.nnz, csr_mat.col_i, csr_mat.row_p);
            mat_->set_dim(csr_mat.num_cols, csr_mat.num_rows);
            mat_->set( csr_mat.data, csr_mat.col, csr_mat.row );        
        }
        else
        {
            throw std::runtime_error("csr::matrix_market_reader::set_csr_matrix: one must read a file before setting a matrix. Call 'read_file' before or 'read_file_2_csr_marix' ");
        }
    }

    void get_matrix_dim(Ord& size_rows_, Ord& size_cols_)
    {
        if(!matrix_file_read)
        {
            throw std::runtime_error("csr::matrix_market_reader::allocate_set_csr_pointers: one must read a file before setting pinters. Call 'read_file' before ");
        } 
        size_cols_ = csr_mat.num_cols;
        size_rows_ = csr_mat.num_rows;
    }
    void allocate_set_csr_pointers(Ord& row_p_, Ord& col_i_, Ord& nnz_, Ord*& row_data_, Ord*& col_data_, T*& values_data_)
    {
        if( (row_data_!=nullptr)||(col_data_!=nullptr)||(values_data_!=nullptr) )
        {
            throw std::runtime_error("csr::matrix_market_reader::allocate_set_csr_pointers: pointers must be 'nullptr' i.e. not set before this call ");
        }
        if(!matrix_file_read)
        {
            throw std::runtime_error("csr::matrix_market_reader::allocate_set_csr_pointers: one must read a file before setting pinters. Call 'read_file' before ");
        }        
        
        row_p_ = csr_mat.row_p;
        col_i_ = csr_mat.col_i;
        nnz_ = csr_mat.nnz;
        row_data_ = (Ord*)malloc(sizeof(Ord)*row_p_);
        if(row_data_ == NULL)
        {
            throw std::runtime_error("csr::matrix_market_reader::allocate_set_csr_pointers: failed to allocate row_data_ memory ");
        }
        col_data_ = (Ord*)malloc(sizeof(Ord)*col_i_);
        if(col_data_ == NULL)
        {
            free(row_data_);
            throw std::runtime_error("csr::matrix_market_reader::allocate_set_csr_pointers: failed to allocate col_data_ memory ");
        }        
        values_data_ = (T*)malloc(sizeof(T)*nnz_);
        if(values_data_ == NULL)
        {
            free(col_data_);
            free(row_data_);
            throw std::runtime_error("csr::matrix_market_reader::allocate_set_csr_pointers: failed to allocate values_data_ memory ");            
        }
        for(Ord j = 0; j<row_p_;j++)
        {
            row_data_[j] = csr_mat.row[j];
        }
        if(col_i_ != nnz_) //why?! to be on the safe side?
        {
            for(Ord j = 0; j<col_i_;j++)
            {
                col_data_[j] = csr_mat.col[j];
            }
            for(Ord j = 0; j<nnz_;j++)
            {
                values_data_[j] = csr_mat.data[j];
            }        
        }
        else
        {
            for(Ord j = 0; j<nnz_;j++)
            {
                col_data_[j] = csr_mat.col[j];
                values_data_[j] = csr_mat.data[j];
            }             
        }

    }

    void read_file(const std::string& file_name_)
    {
        auto start_ch = std::chrono::steady_clock::now();

        std::ifstream fin(file_name_);
        if(!fin.is_open())
        {
            throw std::runtime_error("csr::matrix_market_reader::read_file: error while opening file");
        }
        read_banner(fin);
        if(std::is_same<T, T_real>::value)
        {
            throw std::runtime_error("csr::matrix_market_reader::read_file: real data type used for a complex matrix. This will be fixed later." );
        }        
        Ord num_rows, num_cols, num_entries;
        if (!(fin >> num_rows >> num_cols >> num_entries))
        {
            std::cerr << num_rows << " " << num_cols << " " << num_entries << std::endl;
            throw std::runtime_error("csr::matrix_market_reader::read_file: error while reading sizes");
        }
        print_log("matrix sizes: " + std::to_string(num_rows) + " " + std::to_string(num_cols) + " " + std::to_string(num_entries) );

        coo_matrix coo_mat;
        coo_mat.num_rows = num_rows;
        coo_mat.num_cols = num_cols;
        coo_mat.nnz = num_entries;
        coo_mat.allocate();

        Ord num_entries_read = 0;
        Ord num_entries_ref = num_entries;
        if(num_entries > 10)
        {
           num_entries_ref =  num_entries/10;
        }

        if (banner.type == "pattern") //reading graph matrix
        {
            while(num_entries_read < num_entries && !fin.eof())
            {
                node node_l;
                fin >> node_l.row >> node_l.col;
                node_l.val = T(1);
                coo_mat.data.push_back(node_l);

                num_entries_read++;
                if(num_entries_read%(num_entries_ref) == 0)
                {
                    print_log(std::to_string( static_cast<int>(100.0*num_entries_read/(num_entries))) );
                }
            }

        }         
        else if (banner.type == "real" || banner.type == "integer")
        {
            while( num_entries_read < num_entries && !fin.eof() )
            {
                node node_l;
                fin >> node_l.row >> node_l.col >> node_l.val;
                coo_mat.data.push_back(node_l);
                if(num_entries_read%(num_entries_ref) == 0)
                {
                    print_log(std::to_string( static_cast<int>(100.0*num_entries_read/(num_entries)))+"%" );
                } 
                num_entries_read++;

               
            }
        }
        else if (banner.type == "complex")
        {

            while( num_entries_read < num_entries && !fin.eof() )
            {
                double real_, imag_;
                node node_l;
                fin >> node_l.row >> node_l.col >> real_ >> imag_;
                set_val(node_l.val, real_, imag_);
                coo_mat.data.push_back( node_l );
                num_entries_read++;
                if(num_entries_read%(num_entries_ref) == 0)
                {
                    print_log(std::to_string( static_cast<int>(100.0*num_entries_read/(num_entries)))+"%" );
                }                 
            }
        }        
        else
        {
            throw std::runtime_error("csr::matrix_market_reader::read_file: unsupported matrix data type: " + banner.type );
        }        
        if(num_entries_read != num_entries)
        {
            throw std::runtime_error("csr::matrix_market_reader::read_file: unexpected EOF while reading MatrixMarket entries");
        }
        check_valid_coo_matrix(coo_mat);
        
        //convert to base0 index
        for(Ord j = 0; j < coo_mat.nnz; j++)
        {
            coo_mat.data[j].row -= 1;
            coo_mat.data[j].col -= 1;
        }


        if (banner.symmetry != "general")
        {        
            Ord off_diagonals = 0;

            for (Ord j = 0; j < coo_mat.nnz; j++)
                if(coo_mat.data[j].row != coo_mat.data[j].col)
                    off_diagonals++;

            Ord general_num_entries = coo_mat.nnz + off_diagonals;

            coo_matrix coo_mat_gen;
        
            coo_mat.nnz = general_num_entries;
            coo_mat.data.reserve(general_num_entries);

            if (banner.symmetry == "symmetric")
            {
                Ord nnz = 0;

                for (Ord n = 0; n < num_entries; n++)
                {
                    // // copy entry over
                    // general.row_indices[nnz]    = coo_mat.row_indices[n];
                    // general.column_indices[nnz] = coo_mat.column_indices[n];
                    // general.values[nnz]         = coo_mat.values[n];

                    node node_l;
                    // node_l.row = coo_mat.data.at(n).row;
                    // node_l.col = coo_mat.data.at(n).col;
                    // node_l.val = coo_mat.data.at(n).val;
                    // coo_mat_gen.data.push_back(node_l);
                    nnz++;

                // duplicate off-diagonals
                    if ( coo_mat.data.at(n).row != coo_mat.data.at(n).col )
                    {
                        node_l.row = coo_mat.data.at(n).col;
                        node_l.col = coo_mat.data.at(n).row;
                        node_l.val = coo_mat.data.at(n).val;
                        coo_mat.data.push_back(node_l);
                        nnz++;
                    } 
                }       
            } 
            else if (banner.symmetry == "hermitian")
            {
            throw std::runtime_error("csr::matrix_market_reader::read_file: MatrixMarket I/O does not currently support hermitian matrices");
            //TODO
            } 
            else if (banner.symmetry == "skew-symmetric")
            {
            //TODO
            throw std::runtime_error("csr::matrix_market_reader::read_file: MatrixMarket I/O does not currently support skew-symmetric matrices");
            }            
        }
        fin.close();
        print_log("Reading done. Converting to csr...");
        sort_coo_rows_cols(coo_mat);
        
        csr_mat.nnz = coo_mat.nnz;
        csr_mat.num_cols = coo_mat.num_cols;
        csr_mat.num_rows = coo_mat.num_rows;
        csr_mat.allocate();
        coo_2_csr(coo_mat, csr_mat);
       
        auto finish_ch = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(finish_ch - start_ch).count();

        std::string msg_time = "csr ready. Took " + std::to_string(elapsed_seconds) + " sec.";
        print_log(msg_time);

        matrix_file_read = true;
    }

    void print_coo_matrix(const coo_matrix& coo_mat_)
    {
        
        std::cout << coo_mat_.num_rows << " " << coo_mat_.num_cols << " " << coo_mat_.nnz << std::endl;
        for(int j = 0; j<coo_mat_.nnz;j++)
        {
            std::cout << coo_mat_.data.at(j).row << " " << coo_mat_.data.at(j).col << " " << coo_mat_.data.at(j).val << std::endl;
        }
        
    }
    void print_csr_matrix(const csr_matrix& csr_mat_)
    {
        
        std::cout << csr_mat_.row_p << " " << csr_mat_.col_i << " " << csr_mat_.nnz << std::endl;
        for(int j = 0; j<csr_mat_.nnz;j++)
        {
            std::cout << csr_mat_.data[j] << " ";
        }
        std::cout << std::endl;       

        for(int j = 0; j<csr_mat_.col_i;j++)
        {
            std::cout << csr_mat_.col[j] << " ";
        }
        std::cout << std::endl;
        
        for(int j = 0; j<csr_mat_.row_p;j++)
        {
            std::cout << csr_mat_.row[j] << " ";
        }
        std::cout << std::endl;

        
    }

private:

    void coo_2_csr(const coo_matrix& coo_mat_, csr_matrix& csr_mat_)
    {
        //assume that all raws for csr are set to 0
        for (Ord i = 0; i < coo_mat_.nnz; i++)
        {
            csr_mat_.data[i] = coo_mat_.data.at(i).val;
            csr_mat_.col[i] = coo_mat_.data.at(i).col;
            Ord row_ind = coo_mat_.data.at(i).row;
            csr_mat_.row[row_ind + 1]++; 
        }
        for (int i = 0; i < coo_mat_.num_rows; i++)
        {
            csr_mat_.row[i + 1] += csr_mat_.row[i];
        }
    }

    void sort_coo_rows_cols(coo_matrix& coo_mat_)
    {
        std::stable_sort
        (
            coo_mat_.data.begin(), coo_mat_.data.end(),
            [](const node& a, const node& b)
            {
                if (a.row < b.row) return true;
                if (b.row < a.row) return false;
                if (a.col < b.col) return true;
                if (b.col < a.col) return false;
                return false;
            }
        
        );

    }

    void check_valid_coo_matrix(const coo_matrix& coo_mat_)
    {
        // check validity of row and column index data
        if (coo_mat_.nnz > 0)
        {
            auto min_row_index = *std::min_element(
                coo_mat_.data.begin(), coo_mat_.data.end(),
                [](const node& a, const node& b)
                {
                    if (a.row < b.row)
                    {
                        return true;
                    }
                    else
                    {   
                        return false;
                    }
                }
                );
            auto max_row_index = *std::max_element(
                coo_mat_.data.begin(), coo_mat_.data.end(),
                [](const node& a, const node& b)
                {
                    if (a.row < b.row)
                    {
                        return true;
                    }
                    else
                    {   
                        return false;
                    }
                }
                );            
            auto min_col_index = *std::min_element(
                coo_mat_.data.begin(), coo_mat_.data.end(),
                [](const node& a, const node& b)
                {
                    if (a.col < b.col)
                    {
                        return true;
                    }
                    else
                    {   
                        return false;
                    }
                }
                );   
            auto max_col_index = *std::max_element(
                coo_mat_.data.begin(), coo_mat_.data.end(),
                [](const node& a, const node& b)
                {
                    if (a.col < b.col)
                    {
                        return true;
                    }
                    else
                    {   
                        return false;
                    }
                }
                );

            if (min_row_index.row < 1) throw std::runtime_error("csr::matrix_market_reader::check_valid_coo_matrix: found invalid row index (index < 1)"); 
            if (min_col_index.col < 1) throw std::runtime_error("csr::matrix_market_reader::check_valid_coo_matrix: found invalid column index (index < 1)");
            if (max_row_index.row > coo_mat_.num_rows) throw std::runtime_error("csr::matrix_market_reader::check_valid_coo_matrix: found invalid row index (index > num_rows)");
            if (max_col_index.col > coo_mat_.num_cols) throw std::runtime_error("csr::matrix_market_reader::check_valid_coo_matrix: found invalid column index (index > num_columns)");
        }
    }


    void print_log(const std::string& line_)
    {
        if(verbose)
        {
            std::cout << "csr::mm_reader: " << line_ << std::endl;
        }
    }
    bool matrix_file_read = false;
    bool verbose = false;


    struct matrix_market_banner
    {
        std::string storage;    // "array" or "coordinate"
        std::string symmetry;   // "general", "symmetric", "hermitian", or "skew-symmetric" 
        std::string type;       // "complex", "real", "integer", or "pattern"
        int block_dim_1 = 1;
        int block_dim_2 = 1;
        bool block_matrix = false;
    };
    matrix_market_banner banner;

    void read_banner(std::ifstream& input_stream)
    {
        std::vector<std::string> tokens;
        std::string line;
        std::getline(input_stream, line);
        tokenize(tokens, line);
        if(tokens.size() != 5 || tokens[0] != "%%MatrixMarket" || tokens[1] != "matrix")
        {
            throw std::runtime_error("csr::matrix_market_reader::read_banner: invalid matrix market banner");
        }
        banner.storage  = tokens[2];
        banner.type     = tokens[3];
        banner.symmetry = tokens[4];
        if(banner.storage != "array" && banner.storage != "coordinate")
        {
            throw std::runtime_error("csr::matrix_market_reader::read_banner: invalid MatrixMarket storage format [" + banner.storage + "]");
        }
        if(banner.type != "complex" && banner.type != "real" && banner.type != "integer" && banner.type != "pattern")
        {
            throw std::runtime_error("csr::matrix_market_reader::read_banner: invalid MatrixMarket data type [" + banner.type + "]");
        }
        if(banner.symmetry != "general" && banner.symmetry != "symmetric" && banner.symmetry != "hermitian" && banner.symmetry != "skew-symmetric")
        {
            throw std::runtime_error("csr::matrix_market_reader::read_banner: invalid MatrixMarket symmetry [" + banner.symmetry + "]");
        }

        while (input_stream.peek() == '%')
        {
            try
            {
                std::streampos oldpos = input_stream.tellg();
                std::string comments_line;
                getline(input_stream, comments_line);
                if (comments_line.find(std::string("AMG")) != std::string::npos) 
                {
                    input_stream.seekg(oldpos);
                    std::string buf;
                    input_stream >> buf >> banner.block_dim_1 >> banner.block_dim_2;
                    if(banner.block_dim_1>1 || banner.block_dim_2>1)
                    {
                        banner.block_matrix = true;
                    }
                    while (input_stream.peek() != '\n')
                    {
                        input_stream >> buf;
                    }                    
                    break;
                }
            }
            catch(...)
            {
                throw std::runtime_error("csr::matrix::read_mat: error while searching for the block size");
            }

        }        
        std::string MMbaner = "storage: " + banner.storage + ", symmetry: " + banner.symmetry + ", data type: " + banner.type;
        print_log("MatrixMarket banner read: " + MMbaner);
        if(banner.block_matrix)
        {
            print_log("AMGX banner read: block_dim: " + std::to_string(banner.block_dim_1) + "X" + std::to_string(banner.block_dim_2) );
            std::cerr << "WARNING: the parcing of AMGX block matrices is not currently implemented." << std::endl;
        }

    }  
    void tokenize(std::vector<std::string>& tokens, const std::string& str,                   const std::string& delimiters = "\n\r\t ")
    {
        // Skip delimiters at beginning.
        std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
        // Find first "non-delimiter".
        std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
        while (std::string::npos != pos || std::string::npos != lastPos)
        {
            // Found a token, add it to the vector.
            tokens.push_back(str.substr(lastPos, pos - lastPos));
            // Skip delimiters.  Note the "not_of"
            lastPos = str.find_first_not_of(delimiters, pos);
            // Find next "non-delimiter"
            pos = str.find_first_of(delimiters, lastPos);
        }
    }



};
}
#endif