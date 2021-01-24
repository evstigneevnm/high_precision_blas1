#ifndef __THREADED_DOT_PRODUCT_H__
#define __THREADED_DOT_PRODUCT_H__


#include <cmath>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include <iostream>

template<class T, class T_vec>
class threaded_dot_prod
{
private:
    using accumulators = std::pair< T, T >;
public:
    threaded_dot_prod(size_t vec_size_, int n_ = -1, int use_dot_prod_ = 0, T initial_ = T(0.0)):
    use_dot_prod(use_dot_prod_),
    parts(n_),
    vec_size(vec_size_)
    {
        result = T(0.0);
        sigma.first = T(0.0);
        sigma.second = T(0.0);
        if(parts<1)
        {
            parts = std::thread::hardware_concurrency();
        }
        bounds();
        for(int j=0; j<parts; j++)
        {
            dot_naive.emplace_back(array_bounds[j], array_bounds[j+1]);
            dot_ogita.emplace_back(array_bounds[j], array_bounds[j+1]);
        }
        g_lock_sp = std::make_shared<std::mutex>();

    }
    ~threaded_dot_prod()
    {

    }
private:
    class dot_product_naive
    {
    public:
        dot_product_naive(int begin_, int end_): begin(begin_), end(end_){}
        ~dot_product_naive(){}
        
        void operator ()(const T_vec& x_, const T_vec& y_, T& result_, std::shared_ptr<std::mutex> g_lock_)
        {

            T partial_sum = 0;
            for(int i = begin; i < end; ++i)
            {
                partial_sum += x_[i] * y_[i];
            }
            
            std::lock_guard<std::mutex> lock(*g_lock_);
            result_ = result_ + partial_sum;
        }    
    private:
        int begin;
        int end;

    };

    class dot_product_ogita
    {
    public:
        dot_product_ogita(int begin_, int end_): begin(begin_), end(end_){}
        ~dot_product_ogita(){}
        
        void operator ()(const T_vec& x_, const T_vec& y_, T& result_, accumulators& sigma_, std::shared_ptr<std::mutex> g_lock_)
        {

            T s = T(0.0), c = T(0.0), p = T(0.0);
            pi = T(0.0);
            t = T(0.0);
            for (int j=begin; j<end; j++) 
            {
                p = two_prod(pi, x_[j], y_[j]);
                s = two_sum(t, s, p);
                c = c + pi + t;
            }
            
            std::lock_guard<std::mutex> lock(*g_lock_);
            result_ = two_sum(t, T(result_), s);
            sigma_.first = two_sum(pi, T(sigma_.first), c);
            sigma_.second = T(sigma_.second) + t + pi;

        }    
    private:
        int begin;
        int end;
        mutable T pi = T(0.0);
        mutable T t = T(0.0);

        T two_prod(T &t, T a, T b) const // [1], pdf: 71, 169, 198, 
        {
            T p = a*b;
            t = std::fma(a, b, -p);
            return p;
        }

        T two_sum(T &t, T a, T b) const
        {
            T s = a+b;
            T z = s-a;
            t = a-(s-z)+b-z;
            return s;
        }
    };

public:
    T execute(const T_vec& x_, const T_vec& y_) //const
    {   
        result = T(0.0);
        sigma.first = T(0.0);
        sigma.second = T(0.0);

        std::vector<std::thread> threads;
        threads.reserve(parts);

        for (int j = 0; j < parts; j++) 
        {
            
            if(use_dot_prod == 0)
            {
                threads.push_back( std::thread( std::ref(dot_naive[j]),  std::ref(x_),  std::ref(y_),  std::ref(result), g_lock_sp) );
            }
            else if(use_dot_prod == 1)
            {
                threads.push_back( std::thread( std::ref(dot_ogita[j]),  std::ref(x_),  std::ref(y_),  std::ref(result), std::ref(sigma), g_lock_sp) );
            }
            else
            {
                throw std::logic_error("Incorrect dot product scheme selected");
            }


        }

        for(auto &t : threads)
        {
            if(t.joinable())
                t.join();
        }

        if(use_dot_prod == 0)
        {
            return T(result);
        }
        else
        {
            return T(result) + T(sigma.first) + T(sigma.second);
        }
    }

private:
    T result;
    int use_dot_prod;
    int parts;
    size_t vec_size;
    std::vector<int> array_bounds;
    std::vector<dot_product_naive> dot_naive;
    std::vector<dot_product_ogita> dot_ogita;
    accumulators sigma;
    
    std::shared_ptr<std::mutex> g_lock_sp;


    void bounds()
    {
        array_bounds.reserve(parts+1);
        int delta = vec_size / parts;
        int reminder = vec_size % parts;
        int N1 = 0, N2 = 0;
        array_bounds.push_back(N1);
        for (int j = 0; j < parts; j++) 
        {
            N2 = N1 + delta;
            if (j == parts - 1)
            {
                N2 += reminder;
            }
            array_bounds.push_back(N2);
            N1 = N2;
        }
    }


};


#endif