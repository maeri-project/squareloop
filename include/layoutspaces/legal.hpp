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

#include <vector>

#include "util/numeric.hpp"
#include "util/misc.hpp"
#include "layoutspaces/layoutspace-base.hpp"
#include "compound-config/compound-config.hpp"
#include "layout/layout.hpp"

namespace layoutspace
{

//--------------------------------------------//
//               Legal LayoutSpace             //
//--------------------------------------------//

class Legal : public LayoutSpace
{
 protected:

  // Splits of this layoutspace (used for parallelizing).
  std::vector<Legal*> splits_;
  std::uint64_t split_id_;
  std::uint64_t num_parent_splits_;
  
  // Layout patterns and configurations
  std::vector<std::string> available_patterns_;
  std::vector<std::vector<std::string>> permutation_options_;
  std::vector<std::vector<uint32_t>> blocking_factor_options_;

  // Layout constraint variables
  std::vector<std::pair<std::string, uint64_t>> concordant_temporal_loops_;
  std::vector<std::pair<std::string, uint64_t>> concordant_spatial_loops_;
  bool legal_space_empty_;

  // AuthBlock factor variation tracking
  std::vector<std::tuple<unsigned, unsigned, std::string, uint32_t>> variable_authblock_factors_; // level, dataspace, rank, max_value
  std::vector<uint32_t> authblock_factor_ranges_;

 public:

  //
  // Legal() - Constructor for mapping-based layout creation
  //
  Legal(
    model::Engine::Specs arch_specs,
    const Mapping& mapping,
    layout::Layouts& layout,
    bool skip_init = false);
    
  Legal(const Legal& other) = default;
  ~Legal();

  //------------------------------------------//
  //        Initialization and Setup          // 
  //------------------------------------------//

  void Init(model::Engine::Specs arch_specs, const Mapping& mapping);  

  // Construct a specific layout from the layoutspace.
  std::vector<Status> ConstructLayout(ID layout_id, layout::Layouts* layouts, bool break_on_failure = true) override;

 protected:

  // Helper methods for layout construction
  layout::Layout ConstructBasicLayout(const std::string& target);
  void ApplyDataLayoutPattern(layout::Layout& layout, const std::string& pattern);
  void ApplyPermutationOrder(layout::Layout& layout, const std::vector<std::string>& order);
  void ApplyBlockingFactors(layout::Layout& layout, const std::vector<uint32_t>& factors);

  // Layout constraint methods
  void CreateConcordantLayout(const Mapping& mapping);
  void CreateSpace(model::Engine::Specs arch_specs);
  bool CheckBufferCapacityConstraint(model::Engine::Specs arch_specs, const Mapping& mapping);
};

} // namespace layoutspace 