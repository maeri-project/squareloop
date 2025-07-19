/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <boost/multiprecision/cpp_int.hpp>

#include "util/numeric.hpp"
#include "layout/layout.hpp"
#include "model/engine.hpp"
#include "mapping/mapping.hpp"

using namespace boost::multiprecision;

namespace layoutspace
{

//--------------------------------------------//
//            Layoutspace Dimensions          //
//--------------------------------------------//

enum class Dimension
{
  Intraline,    // Put spatial loops into intraline, the remaining loops will fall in interline.
  Num
};

std::ostream& operator << (std::ostream& out, Dimension d);

//--------------------------------------------//
//               LayoutSpace ID               //
//--------------------------------------------//

typedef CartesianCounter<int(Dimension::Num)> ID;

//--------------------------------------------//
//                   Status                   //
//--------------------------------------------//

struct Status
{
  bool success;
  std::string fail_reason;
};

//--------------------------------------------//
//                 LayoutSpace                //
//--------------------------------------------//

class LayoutSpace
{
 protected:
  model::Engine::Specs arch_specs_;
  const Mapping& mapping_;
  layout::Layouts& layout_;
  
  // layout::LayoutConstraint& layout_constraint_;
  std::array<uint128_t, int(Dimension::Num)> size_;

 public:
  LayoutSpace(model::Engine::Specs arch_specs,
              const Mapping& mapping,
              layout::Layouts& layout) :
      arch_specs_(arch_specs),
      mapping_(mapping),
      layout_(layout),
      size_({})
  {}

  virtual ~LayoutSpace() {}

  virtual std::vector<LayoutSpace*> Split(std::uint64_t num_splits) = 0;

  virtual void InitPruned(uint128_t local_layout_id) = 0;

  virtual std::vector<Status> ConstructLayout(ID layout_id, layout::Layouts* layouts, bool break_on_failure = true) = 0;

  std::vector<Status> ConstructLayout(const uint128_t layout_id, layout::Layouts* layouts, bool break_on_failure = true)
  {
    ID clayout_id(size_);
    clayout_id.Set(layout_id);
    return ConstructLayout(clayout_id, layouts, break_on_failure); 
  }

  uint128_t Size(Dimension dim)
  {
    return size_[int(dim)];
  }

  uint128_t Size()
  {
    uint128_t size = 1;
    for (int i = 0; i < int(Dimension::Num); i++)
    {
      size *= size_[i];
    }
    return size;
  }

  std::array<uint128_t, int(Dimension::Num)> AllSizes()
  {
    return size_;
  }
};

} // namespace layoutspace
