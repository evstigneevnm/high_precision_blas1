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

#ifndef _CUDA_TIMER_EVENT_H__
#define _CUDA_TIMER_EVENT_H__

#include <utils/cuda_safe_call.h>
#include "timer_event.h"

namespace utils
{

struct cuda_timer_event : public timer_event
{
    cudaEvent_t     e;

    cuda_timer_event()
    {
    }
    ~cuda_timer_event()
    {
    }
    virtual void    init()
    {
        CUDA_SAFE_CALL( cudaEventCreate( &e ) );
    }
    virtual void    record()
    {
        cudaEventRecord( e, 0 );

    }
    virtual double  elapsed_time(const timer_event &e0)const
    {
        const cuda_timer_event *cuda_event = dynamic_cast<const cuda_timer_event*>(&e0);
        if (cuda_event == NULL) {
            throw std::logic_error("cuda_timer_event::elapsed_time: try to calc time from different type of timer (non-cuda)");
        }
        float   res;
        cudaEventSynchronize( e );
        cudaEventElapsedTime( &res, cuda_event->e, e );
        return (double)res;
    };
    virtual void    release()
    {
        cudaEventDestroy( e );
    }
};

}

#endif
