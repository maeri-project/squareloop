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

 public:
  // AuthBlock factor variation tracking
  std::vector<std::tuple<unsigned, unsigned, std::string, uint32_t>> variable_authblock_factors_; // level, dataspace, rank, max_value
  std::vector<std::vector<uint32_t>> authblock_factor_ranges_; // stores divisors for each factor


  // Intraline-to-interline conversion factor tracking (new level-based structure)
  struct SplittingOption {
    unsigned dataspace;
    std::string rank;
    uint32_t original_intraline_factor;
    uint32_t splitting_factor;
  };
  std::vector<std::vector<SplittingOption>> splitting_options_per_level_; // [level][option_index]
  std::vector<uint64_t> splitting_choices_per_level_; // number of choices for each level (including "no splitting")

  // Multi-rank splitting option for combinations of ranks
  struct MultiRankSplittingOption {
    unsigned dataspace;
    std::vector<std::string> ranks;  // Multiple ranks involved in the combination
    std::map<std::string, uint32_t> original_intraline_factors;  // Original factors for each rank
    std::map<std::string, uint32_t> splitting_factors;  // Splitting factors for each rank
    uint64_t total_reduction;  // Total reduction in intraline size from this combination
  };
  std::vector<std::vector<MultiRankSplittingOption>> multi_rank_splitting_options_per_level_; // [level][option_index]

  // Cross-dataspace multi-rank splitting option for combinations of ranks across multiple dataspaces
  struct CrossDataspaceMultiRankSplittingOption {
    std::vector<std::string> ranks;  // Multiple ranks involved (with dataspace prefixes like "DS0_K", "DS1_H")
    std::map<std::string, uint32_t> original_intraline_factors;  // Original factors for each rank
    std::map<std::string, uint32_t> splitting_factors;  // Splitting factors for each rank
    std::map<std::string, unsigned> rank_to_dataspace;  // Map rank to its dataspace index
    uint64_t total_reduction;  // Total reduction in intraline size from this combination
  };
  std::vector<std::vector<CrossDataspaceMultiRankSplittingOption>> cross_dataspace_multi_rank_splitting_options_per_level_; // [level][option_index]

  // Track which levels require splitting (where intraline_size > line_capacity)
  std::vector<bool> level_requires_splitting_; // [level] -> true if splitting is mandatory

  // Interline-to-intraline packing factor tracking (for unused line capacity)
  // Restructured to support single-rank-per-level packing
  struct PackingOption {
    unsigned dataspace;
    std::string rank;
    uint32_t original_interline_factor;
    uint32_t packing_factor;
  };
  
  // Multi-rank packing option for combinations of ranks within a single dataspace
  struct MultiRankPackingOption {
    unsigned dataspace;
    std::vector<std::string> ranks;  // Multiple ranks involved in the combination
    std::map<std::string, uint32_t> original_interline_factors;  // Original factors for each rank
    std::map<std::string, uint32_t> packing_factors;  // Packing factors for each rank
    uint64_t total_packing;  // Total packing factor applied
  };
  
  // Cross-dataspace multi-rank packing option for combinations of ranks across multiple dataspaces
  struct CrossDataspaceMultiRankPackingOption {
    std::vector<std::string> ranks;  // Multiple ranks involved (with dataspace prefixes like "DS0_K", "DS1_H")
    std::map<std::string, uint32_t> original_interline_factors;  // Original factors for each rank
    std::map<std::string, uint32_t> packing_factors;  // Packing factors for each rank
    std::map<std::string, unsigned> rank_to_dataspace;  // Map rank to its dataspace index
    uint64_t total_packing;  // Total packing factor applied
  };
  
  // Packing choices organized by storage level
  // Each level can choose to pack exactly one rank (or no packing)
  std::vector<std::vector<PackingOption>> packing_options_per_level_; // [level][option_index]
  std::vector<std::vector<MultiRankPackingOption>> multi_rank_packing_options_per_level_; // [level][option_index]
  std::vector<std::vector<CrossDataspaceMultiRankPackingOption>> cross_dataspace_multi_rank_packing_options_per_level_; // [level][option_index]
  std::vector<uint64_t> packing_choices_per_level_; // number of choices for each level (including "no packing")
  unsigned num_storage_levels;
  unsigned num_data_spaces;

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

  // Override the pure virtual function from LayoutSpace base class
  std::vector<Status> ConstructLayout(ID layout_id, layout::Layouts* layouts, Mapping mapping, bool break_on_failure = true) override;

  // Construct a specific layout using separate IDs for all three design spaces.
  std::vector<Status> ConstructLayout(uint64_t layout_splitting_id, uint64_t layout_packing_id, uint64_t layout_auth_id, layout::Layouts* layouts, Mapping mapping, bool break_on_failure = true);

 protected:

  // Helper methods for layout construction
  layout::Layout ConstructBasicLayout(const std::string& target);
  void ApplyDataLayoutPattern(layout::Layout& layout, const std::string& pattern);
  void ApplyPermutationOrder(layout::Layout& layout, const std::vector<std::string>& order);
  void ApplyBlockingFactors(layout::Layout& layout, const std::vector<uint32_t>& factors);

  // Layout constraint methods
  void CreateConcordantLayout(const Mapping& mapping);
  void CreateSpace(model::Engine::Specs arch_specs);
  void CreateIntralineFactorSpace(model::Engine::Specs arch_specs, const Mapping& mapping);
  void CreateAuthSpace(model::Engine::Specs arch_specs);

  // Helper methods for multi-rank splitting
  std::vector<std::vector<std::string>> GenerateRankCombinations(const std::vector<std::string>& ranks, size_t max_combo_size = 3);
  bool TestMultiRankSplittingWithCandidates(unsigned lvl, unsigned ds_idx, const std::vector<std::string>& rank_combination,
                                           const std::map<std::string, std::vector<uint32_t>>& candidate_factors_per_rank,
                                           const std::vector<std::vector<std::uint64_t>>& intraline_size_per_ds,
                                           uint64_t line_capacity, MultiRankSplittingOption& option);

  // Helper method for cross-dataspace multi-rank splitting
  bool TestCrossDataspaceMultiRankSplittingWithCandidates(unsigned lvl, const std::vector<std::string>& rank_combination,
                                                         const std::map<std::string, std::vector<uint32_t>>& candidate_factors_per_rank,
                                                         const std::map<std::string, std::pair<unsigned, uint32_t>>& rank_to_dataspace_and_original_factor,
                                                         const std::vector<std::vector<std::uint64_t>>& intraline_size_per_ds,
                                                         uint64_t line_capacity, CrossDataspaceMultiRankSplittingOption& option);

  // Helper methods for multi-rank packing
  bool TestMultiRankPackingWithCandidates(unsigned lvl, unsigned ds_idx, const std::vector<std::string>& rank_combination,
                                         const std::map<std::string, std::vector<uint32_t>>& candidate_factors_per_rank,
                                         const std::vector<std::vector<std::uint64_t>>& intraline_size_per_ds,
                                         uint64_t line_capacity, MultiRankPackingOption& option);

  bool TestCrossDataspaceMultiRankPackingWithCandidates(unsigned lvl, const std::vector<std::string>& rank_combination,
                                                       const std::map<std::string, std::vector<uint32_t>>& candidate_factors_per_rank,
                                                       const std::map<std::string, std::pair<unsigned, uint32_t>>& rank_to_dataspace_and_original_factor,
                                                       const std::vector<std::vector<std::uint64_t>>& intraline_size_per_ds,
                                                       uint64_t line_capacity, CrossDataspaceMultiRankPackingOption& option);
};

} // namespace layoutspace 