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

#ifndef __LOGGED_OBJ_BASE_H__
#define __LOGGED_OBJ_BASE_H__

#include <cstdarg>
#include <string>

namespace utils
{

template<class Log>
class logged_obj_base
{
protected:
    Log             *log_;
    int             obj_log_lev_;
    std::string     log_msg_prefix_;
public:
    logged_obj_base(Log *log__ = NULL, int obj_log_lev__ = 0, const std::string &log_msg_prefix__ = "") : 
        log_(log__),obj_log_lev_(obj_log_lev__),log_msg_prefix_(log_msg_prefix__) {}

    void                set_log(Log *log__) { log_ = log__; }
    Log                 *get_log()const { return log_; }
    void                set_obj_log_lev(int obj_log_lev__) { obj_log_lev_ = obj_log_lev__; }
    int                 get_obj_log_lev()const { return obj_log_lev_; }
    void                set_log_msg_prefix(const std::string &log_msg_prefix__) 
    { 
        log_msg_prefix_ = log_msg_prefix__; 
    }
    const std::string   &get_log_msg_prefix()const { return log_msg_prefix_; }

    void info(const std::string &s, int log_lev_ = 1)const
    {
        if (log_ != NULL) log_->info(log_msg_prefix_ + s, obj_log_lev_ + log_lev_);
    }
    void info_all(const std::string &s, int log_lev_ = 1)const
    {
        if (log_ != NULL) log_->info_all(log_msg_prefix_ + s, obj_log_lev_ + log_lev_);
    }
    void warning(const std::string &s, int log_lev_ = 1)const
    {
        if (log_ != NULL) log_->warning(log_msg_prefix_ + s, obj_log_lev_ + log_lev_);
    }
    void error(const std::string &s, int log_lev_ = 1)const
    {
        if (log_ != NULL) log_->error(log_msg_prefix_ + s, obj_log_lev_ + log_lev_);
    }

    #define LOGGED_OBJ_BASE__FORMATTED_OUT__(METHOD_NAME, LOG_LEV)              \
        if (log_ == NULL) return;                                               \
        va_list arguments;                                                      \
        va_start ( arguments, s );                                              \
        log_->METHOD_NAME(obj_log_lev_, log_msg_prefix_ + s, arguments);        \
        va_end ( arguments );    
    void info_f(int log_lev_, const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_info_f, obj_log_lev_ + log_lev_)
    }
    void info_all_f(int log_lev_, const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_info_all_f, obj_log_lev_ + log_lev_)
    }
    void warning_f(int log_lev_, const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_warning_f, obj_log_lev_ + log_lev_)
    }
    void error_f(int log_lev_, const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_error_f, obj_log_lev_ + log_lev_)
    }
    void info_f(const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_info_f, obj_log_lev_)
    }
    void info_all_f(const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_info_all_f, obj_log_lev_)
    }
    void warning_f(const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_warning_f, obj_log_lev_)
    }
    void error_f(const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_error_f, obj_log_lev_)
    }
    #undef LOGGED_OBJ_BASE__FORMATTED_OUT__
};

}

#endif 