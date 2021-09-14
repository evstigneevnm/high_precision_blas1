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

#ifndef __STATIC_ASSERT_H__
#define __STATIC_ASSERT_H__

//unforunatly under nvcc this STATIC_ASSERT gives unreadable results (msg is not written and 'STATIC_ASSERT' is not written)
//seems that similar problems has even thrust static_assert
//on the other hand nvcc at least gives number lines of callers where assert was failed

//these with __ are not supposed to be used (just intermediate help macros)
#define __STATIC_ASSERT__CTASTR2(pre,post) pre ## post
#define __STATIC_ASSERT__CTASTR(pre,post) __STATIC_ASSERT__CTASTR2(pre,post)
//no line append becuase compiler seems to give it by himself
//__COUNTER__ requiers gcc at least 4.3; works on cl at least in VS2008 (did not tested former versions)
//ISSUE may be make more 'stupid' realisations for earlier versions of compilers
#define STATIC_ASSERT(cond,msg) \
    typedef struct { int __STATIC_ASSERT__CTASTR(STATIC_ASSERTION_FAILED_,msg) : !!(cond); } \
    __STATIC_ASSERT__CTASTR(STATIC_ASSERTION_FAILED_,__COUNTER__)

#endif