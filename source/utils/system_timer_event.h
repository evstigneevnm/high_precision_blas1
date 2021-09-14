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

#ifndef _SYSTEM_TIMER_EVENT_H__
#define _SYSTEM_TIMER_EVENT_H__

//TODO windows realization

#include <sys/time.h>
#include <unistd.h>
#include "timer_event.h"

namespace utils
{

struct system_timer_event : public timer_event
{
    struct timeval tv;

    system_timer_event()
    {
    }
    ~system_timer_event()
    {
    }
    virtual void    init()
    {
    }
    virtual void    record()
    {
        gettimeofday(&tv, NULL);
    }
    virtual double  elapsed_time(const timer_event &e0)const
    {
        const system_timer_event *event = dynamic_cast<const system_timer_event*>(&e0);
        if (event == NULL) {
            throw std::logic_error("system_timer_event::elapsed_time: try to calc time from different type of timer");
        }
        double  res;
        long    seconds, useconds; 
        seconds  = tv.tv_sec  - event->tv.tv_sec;
        useconds = tv.tv_usec - event->tv.tv_usec;
        res = seconds*1000. + useconds/1000.0;
        return res;
    };
    virtual void    release()
    {
    }
};

}

#endif
