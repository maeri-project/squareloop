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

#include "layoutspaces/legal.hpp"
#include <set>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <functional>
// #define DEBUG_CONCORDANT_LAYOUT
// #define DEBUG_BUFFER_CAPACITY_CONSTRAINT
// #define DEBUG_CONSTRUCTION_LAYOUT
namespace layoutspace
{

//------------------------------------------//
//        Helper Functions                  //
//------------------------------------------//

// Helper function to find all divisors of a number
std::vector<uint32_t> FindDivisors(uint32_t n)
{
  std::vector<uint32_t> divisors;
  for (uint32_t i = 1; i <= n; ++i)
  {
    if (n % i == 0)
    {
      divisors.push_back(i);
    }
  }
  return divisors;
}

// Helper function to generate combinations of ranks for multi-rank splitting
std::vector<std::vector<std::string>> Legal::GenerateRankCombinations(const std::vector<std::string>& ranks, size_t max_combo_size)
{
  std::vector<std::vector<std::string>> combinations;

  // Generate combinations of size 2 to max_combo_size (limited by ranks.size())
  for (size_t combo_size = 2; combo_size <= std::min(max_combo_size, ranks.size()); combo_size++)
  {
    // Generate all combinations of combo_size from ranks
    std::vector<bool> selector(ranks.size());
    std::fill(selector.begin(), selector.begin() + combo_size, true);

    do
    {
      std::vector<std::string> combination;
      for (size_t i = 0; i < ranks.size(); ++i)
      {
        if (selector[i])
        {
          combination.push_back(ranks[i]);
        }
      }
      combinations.push_back(combination);
    } while (std::prev_permutation(selector.begin(), selector.end()));
  }

  return combinations;
}

// Helper function to test multi-rank splitting using pre-computed candidate factors
bool Legal::TestMultiRankSplittingWithCandidates(unsigned lvl, unsigned ds_idx, const std::vector<std::string>& rank_combination,
                                                 const std::map<std::string, std::vector<uint32_t>>& candidate_factors_per_rank,
                                                 const std::vector<std::vector<std::uint64_t>>& intraline_size_per_ds,
                                                 uint64_t line_capacity, MultiRankSplittingOption& option)
{
  auto& intraline_nest = layout_.at(lvl).intraline.at(ds_idx);
  
  // Initialize the option
  option.dataspace = ds_idx;
  option.ranks = rank_combination;
  option.original_intraline_factors.clear();
  option.splitting_factors.clear();
  option.total_reduction = 1;
  
  // Get original factors and candidate splitting factors for each rank in the combination
  std::vector<std::vector<uint32_t>> candidate_factors_list;
  for (const auto& rank : rank_combination) {
    uint32_t original_factor = (intraline_nest.factors.find(rank) != intraline_nest.factors.end()
                               ? intraline_nest.factors.at(rank) : 1);
    option.original_intraline_factors[rank] = original_factor;
    
    // Get candidate factors for this rank
    auto candidates_it = candidate_factors_per_rank.find(rank);
    if (candidates_it == candidate_factors_per_rank.end()) {
      return false; // No candidate factors for this rank
    }
    candidate_factors_list.push_back(candidates_it->second);
  }
  
  // Calculate current intraline size for this dataspace
  uint64_t current_dataspace_intraline_size = intraline_size_per_ds[lvl][ds_idx];
  
  // Generate all combinations of candidate factors using nested loops
  // This is a more comprehensive approach than the equal distribution method
  std::function<bool(size_t, std::vector<uint32_t>&, uint64_t)> try_combinations = 
    [&](size_t rank_idx, std::vector<uint32_t>& current_factors, uint64_t accumulated_reduction) -> bool {
    
    if (rank_idx == rank_combination.size()) {
      // All ranks have been assigned factors, test if this combination works
      // Calculate the new intraline size for the split dataspace
      uint64_t new_dataspace_intraline_size = current_dataspace_intraline_size / accumulated_reduction;
      
      // Calculate total intraline size across ALL dataspaces after applying this split
      uint64_t total_new_intraline_size = 0;
      for (unsigned ds_idx_check = 0; ds_idx_check < intraline_size_per_ds[lvl].size(); ds_idx_check++) {
        if (ds_idx_check == ds_idx) {
          total_new_intraline_size += new_dataspace_intraline_size;
        } else {
          total_new_intraline_size += intraline_size_per_ds[lvl][ds_idx_check];
        }
      }
      
      if (total_new_intraline_size <= line_capacity) {
        // This combination works - store it in the option
        option.total_reduction = accumulated_reduction;
        for (size_t i = 0; i < rank_combination.size(); i++) {
          option.splitting_factors[rank_combination[i]] = current_factors[i];
        }
        return true;
      }
      return false;
    }
    
    // Try each candidate factor for the current rank
    const auto& rank = rank_combination[rank_idx];
    const auto& factors = candidate_factors_list[rank_idx];
    uint32_t original_factor = option.original_intraline_factors.at(rank);
    
    for (uint32_t factor : factors) {
      // Check if this factor is valid (i.e., divides the original factor)
      if (original_factor % factor == 0) {
        current_factors[rank_idx] = factor;
        uint64_t new_accumulated_reduction = accumulated_reduction * factor;
        
        // Recursive call for next rank
        if (try_combinations(rank_idx + 1, current_factors, new_accumulated_reduction)) {
          return true; // Found a valid combination
        }
      }
    }
    
    return false; // No valid combination found with current prefix
  };
  
  // Start the recursive combination testing
  std::vector<uint32_t> current_factors(rank_combination.size());
  return try_combinations(0, current_factors, 1);
}

// Helper function to test cross-dataspace multi-rank splitting using pre-computed candidate factors
bool Legal::TestCrossDataspaceMultiRankSplittingWithCandidates(unsigned lvl, const std::vector<std::string>& rank_combination,
                                                               const std::map<std::string, std::vector<uint32_t>>& candidate_factors_per_rank,
                                                               const std::map<std::string, std::pair<unsigned, uint32_t>>& rank_to_dataspace_and_original_factor,
                                                               const std::vector<std::vector<std::uint64_t>>& intraline_size_per_ds,
                                                               uint64_t line_capacity, CrossDataspaceMultiRankSplittingOption& option)
{
  // Initialize the option
  option.ranks = rank_combination;
  option.original_intraline_factors.clear();
  option.splitting_factors.clear();
  option.rank_to_dataspace.clear();
  option.total_reduction = 1;
  
  // Get original factors and candidate splitting factors for each rank in the combination
  std::vector<std::vector<uint32_t>> candidate_factors_list;
  for (const auto& unique_rank : rank_combination) {
    auto dataspace_and_factor_it = rank_to_dataspace_and_original_factor.find(unique_rank);
    if (dataspace_and_factor_it == rank_to_dataspace_and_original_factor.end()) {
      return false; // Rank not found
    }
    
    unsigned dataspace_idx = dataspace_and_factor_it->second.first;
    uint32_t original_factor = dataspace_and_factor_it->second.second;
    
    option.original_intraline_factors[unique_rank] = original_factor;
    option.rank_to_dataspace[unique_rank] = dataspace_idx;
    
    // Get candidate factors for this rank
    auto candidates_it = candidate_factors_per_rank.find(unique_rank);
    if (candidates_it == candidate_factors_per_rank.end()) {
      return false; // No candidate factors for this rank
    }
    candidate_factors_list.push_back(candidates_it->second);
  }
  
  // Calculate current total intraline size across all dataspaces
  uint64_t current_total_intraline_size = 0;
  for (const auto& dataspace_size : intraline_size_per_ds[lvl]) {
    current_total_intraline_size += dataspace_size;
  }
  
  // Generate all combinations of candidate factors using nested loops
  std::function<bool(size_t, std::vector<uint32_t>&, std::map<unsigned, uint64_t>&)> try_combinations = 
    [&](size_t rank_idx, std::vector<uint32_t>& current_factors, std::map<unsigned, uint64_t>& accumulated_reductions_per_dataspace) -> bool {
    
    if (rank_idx == rank_combination.size()) {
      // All ranks have been assigned factors, test if this combination works
      // Calculate the new total intraline size after applying splits to all affected dataspaces
      uint64_t new_total_intraline_size = 0;
      
      for (unsigned ds_idx = 0; ds_idx < intraline_size_per_ds[lvl].size(); ds_idx++) {
        uint64_t dataspace_reduction = accumulated_reductions_per_dataspace.find(ds_idx) != accumulated_reductions_per_dataspace.end() 
                                     ? accumulated_reductions_per_dataspace[ds_idx] : 1;
        new_total_intraline_size += intraline_size_per_ds[lvl][ds_idx] / dataspace_reduction;
      }
      
      if (new_total_intraline_size <= line_capacity) {
                          // This combination works - store it in the option
         uint64_t total_reduction = 1;
         for (size_t i = 0; i < rank_combination.size(); i++) {
           option.splitting_factors[rank_combination[i]] = current_factors[i];
           total_reduction *= current_factors[i];
         }
         option.total_reduction = total_reduction;
        return true;
      }
      return false;
    }
    
    // Try each candidate factor for the current rank
    const auto& unique_rank = rank_combination[rank_idx];
    const auto& factors = candidate_factors_list[rank_idx];
    uint32_t original_factor = option.original_intraline_factors.at(unique_rank);
    unsigned dataspace_idx = option.rank_to_dataspace.at(unique_rank);
    
    for (uint32_t factor : factors) {
      // Check if this factor is valid (i.e., divides the original factor)
      if (original_factor % factor == 0) {
        current_factors[rank_idx] = factor;
        
        // Update accumulated reduction for this dataspace
        std::map<unsigned, uint64_t> new_accumulated_reductions = accumulated_reductions_per_dataspace;
        if (new_accumulated_reductions.find(dataspace_idx) == new_accumulated_reductions.end()) {
          new_accumulated_reductions[dataspace_idx] = factor;
        } else {
          new_accumulated_reductions[dataspace_idx] *= factor;
        }
        
        // Recursive call for next rank
        if (try_combinations(rank_idx + 1, current_factors, new_accumulated_reductions)) {
          return true; // Found a valid combination
        }
      }
    }
    
    return false; // No valid combination found with current prefix
  };
  
  // Start the recursive combination testing
  std::vector<uint32_t> current_factors(rank_combination.size());
  std::map<unsigned, uint64_t> accumulated_reductions_per_dataspace;
  return try_combinations(0, current_factors, accumulated_reductions_per_dataspace);
}

// Helper function to test multi-rank packing using pre-computed candidate factors
bool Legal::TestMultiRankPackingWithCandidates(unsigned lvl, unsigned ds_idx, const std::vector<std::string>& rank_combination,
                                               const std::map<std::string, std::vector<uint32_t>>& candidate_factors_per_rank,
                                               const std::vector<std::vector<std::uint64_t>>& intraline_size_per_ds,
                                               uint64_t line_capacity, MultiRankPackingOption& option)
{
  auto& interline_nest = layout_.at(lvl).interline.at(ds_idx);
  
  // Initialize the option
  option.dataspace = ds_idx;
  option.ranks = rank_combination;
  option.original_interline_factors.clear();
  option.packing_factors.clear();
  option.total_packing = 1;
  
  // Get original factors and candidate packing factors for each rank in the combination
  std::vector<std::vector<uint32_t>> candidate_factors_list;
  for (const auto& rank : rank_combination) {
    uint32_t original_factor = (interline_nest.factors.find(rank) != interline_nest.factors.end()
                               ? interline_nest.factors.at(rank) : 1);
    option.original_interline_factors[rank] = original_factor;
    
    // Get candidate factors for this rank
    auto candidates_it = candidate_factors_per_rank.find(rank);
    if (candidates_it == candidate_factors_per_rank.end()) {
      return false; // No candidate factors for this rank
    }
    candidate_factors_list.push_back(candidates_it->second);
  }
  
  // Calculate current total intraline size across all dataspaces
  uint64_t current_total_intraline_size = 0;
  for (const auto& dataspace_size : intraline_size_per_ds[lvl]) {
    current_total_intraline_size += dataspace_size;
  }
  
  // Generate all combinations of candidate factors using nested loops
  std::function<bool(size_t, std::vector<uint32_t>&, uint64_t)> try_combinations = 
    [&](size_t rank_idx, std::vector<uint32_t>& current_factors, uint64_t accumulated_packing) -> bool {
    
    if (rank_idx == rank_combination.size()) {
      // All ranks have been assigned factors, test if this combination works
      // Calculate the new total intraline size after applying packing to this dataspace
      uint64_t new_total_intraline_size = 0;
      
      for (unsigned ds_idx_check = 0; ds_idx_check < intraline_size_per_ds[lvl].size(); ds_idx_check++) {
        if (ds_idx_check == ds_idx) {
          new_total_intraline_size += intraline_size_per_ds[lvl][ds_idx_check] * accumulated_packing;
        } else {
          new_total_intraline_size += intraline_size_per_ds[lvl][ds_idx_check];
        }
      }
      
      if (new_total_intraline_size <= line_capacity) {
        // This combination works - store it in the option
        option.total_packing = accumulated_packing;
        for (size_t i = 0; i < rank_combination.size(); i++) {
          option.packing_factors[rank_combination[i]] = current_factors[i];
        }
        return true;
      }
      return false;
    }
    
    // Try each candidate factor for the current rank
    const auto& rank = rank_combination[rank_idx];
    const auto& factors = candidate_factors_list[rank_idx];
    uint32_t original_factor = option.original_interline_factors.at(rank);
    
    for (uint32_t factor : factors) {
      // Check if this factor is valid (i.e., divides the original factor)
      if (original_factor % factor == 0) {
        current_factors[rank_idx] = factor;
        uint64_t new_accumulated_packing = accumulated_packing * factor;
        
        // Recursive call for next rank
        if (try_combinations(rank_idx + 1, current_factors, new_accumulated_packing)) {
          return true; // Found a valid combination
        }
      }
    }
    
    return false; // No valid combination found with current prefix
  };
  
  // Start the recursive combination testing
  std::vector<uint32_t> current_factors(rank_combination.size());
  return try_combinations(0, current_factors, 1);
}

// Helper function to test cross-dataspace multi-rank packing using pre-computed candidate factors
bool Legal::TestCrossDataspaceMultiRankPackingWithCandidates(unsigned lvl, const std::vector<std::string>& rank_combination,
                                                             const std::map<std::string, std::vector<uint32_t>>& candidate_factors_per_rank,
                                                             const std::map<std::string, std::pair<unsigned, uint32_t>>& rank_to_dataspace_and_original_factor,
                                                             const std::vector<std::vector<std::uint64_t>>& intraline_size_per_ds,
                                                             uint64_t line_capacity, CrossDataspaceMultiRankPackingOption& option)
{
  // Initialize the option
  option.ranks = rank_combination;
  option.original_interline_factors.clear();
  option.packing_factors.clear();
  option.rank_to_dataspace.clear();
  option.total_packing = 1;
  
  // Get original factors and candidate packing factors for each rank in the combination
  std::vector<std::vector<uint32_t>> candidate_factors_list;
  for (const auto& unique_rank : rank_combination) {
    auto dataspace_and_factor_it = rank_to_dataspace_and_original_factor.find(unique_rank);
    if (dataspace_and_factor_it == rank_to_dataspace_and_original_factor.end()) {
      return false; // Rank not found
    }
    
    unsigned dataspace_idx = dataspace_and_factor_it->second.first;
    uint32_t original_factor = dataspace_and_factor_it->second.second;
    
    option.original_interline_factors[unique_rank] = original_factor;
    option.rank_to_dataspace[unique_rank] = dataspace_idx;
    
    // Get candidate factors for this rank
    auto candidates_it = candidate_factors_per_rank.find(unique_rank);
    if (candidates_it == candidate_factors_per_rank.end()) {
      return false; // No candidate factors for this rank
    }
    candidate_factors_list.push_back(candidates_it->second);
  }
  
  // Calculate current total intraline size across all dataspaces
  uint64_t current_total_intraline_size = 0;
  for (const auto& dataspace_size : intraline_size_per_ds[lvl]) {
    current_total_intraline_size += dataspace_size;
  }
  
  // Generate all combinations of candidate factors using nested loops
  std::function<bool(size_t, std::vector<uint32_t>&, std::map<unsigned, uint64_t>&)> try_combinations = 
    [&](size_t rank_idx, std::vector<uint32_t>& current_factors, std::map<unsigned, uint64_t>& accumulated_packing_per_dataspace) -> bool {
    
    if (rank_idx == rank_combination.size()) {
      // All ranks have been assigned factors, test if this combination works
      // Calculate the new total intraline size after applying packing to all affected dataspaces
      uint64_t new_total_intraline_size = 0;
      
      for (unsigned ds_idx = 0; ds_idx < intraline_size_per_ds[lvl].size(); ds_idx++) {
        uint64_t dataspace_packing = accumulated_packing_per_dataspace.find(ds_idx) != accumulated_packing_per_dataspace.end() 
                                    ? accumulated_packing_per_dataspace[ds_idx] : 1;
        new_total_intraline_size += intraline_size_per_ds[lvl][ds_idx] * dataspace_packing;
      }
      
      if (new_total_intraline_size <= line_capacity) {
        // This combination works - store it in the option
        uint64_t total_packing = 1;
        for (size_t i = 0; i < rank_combination.size(); i++) {
          option.packing_factors[rank_combination[i]] = current_factors[i];
          total_packing *= current_factors[i];
        }
        option.total_packing = total_packing;
        return true;
      }
      return false;
    }
    
    // Try each candidate factor for the current rank
    const auto& unique_rank = rank_combination[rank_idx];
    const auto& factors = candidate_factors_list[rank_idx];
    uint32_t original_factor = option.original_interline_factors.at(unique_rank);
    unsigned dataspace_idx = option.rank_to_dataspace.at(unique_rank);
    
    for (uint32_t factor : factors) {
      // Check if this factor is valid (i.e., divides the original factor)
      if (original_factor % factor == 0) {
        current_factors[rank_idx] = factor;
        
        // Update accumulated packing for this dataspace
        std::map<unsigned, uint64_t> new_accumulated_packing = accumulated_packing_per_dataspace;
        if (new_accumulated_packing.find(dataspace_idx) == new_accumulated_packing.end()) {
          new_accumulated_packing[dataspace_idx] = factor;
        } else {
          new_accumulated_packing[dataspace_idx] *= factor;
        }
        
        // Recursive call for next rank
        if (try_combinations(rank_idx + 1, current_factors, new_accumulated_packing)) {
          return true; // Found a valid combination
        }
      }
    }
    
    return false; // No valid combination found with current prefix
  };
  
  // Start the recursive combination testing
  std::vector<uint32_t> current_factors(rank_combination.size());
  std::map<unsigned, uint64_t> accumulated_packing_per_dataspace;
  return try_combinations(0, current_factors, accumulated_packing_per_dataspace);
}

//------------------------------------------//
//        Legal LayoutSpace                 //
//------------------------------------------//

//
// Legal() - Constructor with basic initialization
//
Legal::Legal(
  model::Engine::Specs arch_specs,
  const Mapping& mapping,
  layout::Layouts& layout,
  bool skip_init) :
    LayoutSpace(arch_specs, mapping, layout)
{
  layout::PrintOverallLayout(layout);
  (void)skip_init; // Suppress unused parameter warning
  num_storage_levels = mapping.loop_nest.storage_tiling_boundaries.size();
  num_data_spaces = layout_.at(0).intraline.size();

  // Step 0: Initialize the data bypass logic
  Init(arch_specs, mapping);

  // Step 1: Create concordant layout from mapping
  CreateConcordantLayout(mapping);

  // Step 2: Create design spaces for layout optimization
  CreateIntralineFactorSpace(arch_specs, mapping);

  // Step 3: Create AuthSpace
  CreateAuthSpace(arch_specs);
}

Legal::~Legal()
{
  for (auto split : splits_)
  {
    delete split;
  }
  splits_.clear();
}

//------------------------------------------//
//        Initialization and Setup          //
//------------------------------------------//

//
// Init() - called by constructor or derived classes.
//
void Legal::Init(model::Engine::Specs arch_specs,
  const Mapping& mapping)
{
  // Initialize the storage level capacity vectors
  storage_level_total_capacity.resize(num_storage_levels, 0);
  storage_level_line_capacity.resize(num_storage_levels, 0);
  
  // Iterate through each storage level to extract capacity information
  for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++)
  {
    auto storage_level_specs = arch_specs.topology.GetStorageLevel(storage_level);

    // Extract total capacity
    std::uint64_t total_capacity = 0;
    if (storage_level_specs->size.IsSpecified())
    {
      total_capacity = storage_level_specs->size.Get();
    }
    else
    {
      std::cout << "    WARNING: Storage level " << storage_level
                << " (" << storage_level_specs->name.Get() << ") has unspecified size, treating as infinite" << std::endl;
      total_capacity = std::numeric_limits<uint64_t>::max();
    }

    // Determine line capacity (elements that can be accessed in parallel)
    std::uint64_t line_capacity = 0;
    if (storage_level_specs->block_size.IsSpecified())
    {
      line_capacity = storage_level_specs->block_size.Get();
    }
    else
    {
      // Fallback to bandwidth if block size not specified
      double read_bandwidth = storage_level_specs->read_bandwidth.IsSpecified() ? 
                              storage_level_specs->read_bandwidth.Get() : 0.0;
      double write_bandwidth = storage_level_specs->write_bandwidth.IsSpecified() ? 
                               storage_level_specs->write_bandwidth.Get() : 0.0;
      line_capacity = static_cast<std::uint64_t>(std::max(read_bandwidth, write_bandwidth));
    }

    // Store capacity values (with safe casting)
    storage_level_total_capacity[storage_level] = (total_capacity > std::numeric_limits<uint32_t>::max()) ?
                                                  std::numeric_limits<uint32_t>::max() :
                                                  static_cast<std::uint32_t>(total_capacity);
    storage_level_line_capacity[storage_level] = (line_capacity > std::numeric_limits<uint32_t>::max()) ?
                                                 std::numeric_limits<uint32_t>::max() :
                                                 static_cast<std::uint32_t>(line_capacity);

#ifdef DEBUG_BUFFER_CAPACITY_CONSTRAINT
    std::cout << "    Storage Level " << storage_level
              << " (" << storage_level_specs->name.Get() << "):" << std::endl;
    std::cout << "      Total capacity: " << total_capacity << " elements" << std::endl;
    std::cout << "      Line capacity: " << line_capacity << " elements/cycle" << std::endl;
#endif
  }

  (void)mapping; // Suppress unused parameter warning
}

//
// ConstructLayout() - Original version with ID parameter (delegates to three-parameter version)
//
std::vector<Status> Legal::ConstructLayout(ID layout_id, layout::Layouts* layouts, Mapping mapping, bool break_on_failure)
{
  (void)break_on_failure; // Suppress unused parameter warning

  // This function delegates to the three-parameter version with default values
  // Convert ID to uint64_t
  uint128_t layout_int = layout_id.Integer();

  // Check if the layout ID fits in 64-bit range
  if (layout_int > std::numeric_limits<std::uint64_t>::max())
  {
    Status error_status;
    error_status.success = false;
    error_status.fail_reason = "Layout ID exceeds 64-bit range";
    return {error_status};
  }

  // Safe cast to 64-bit
  std::uint64_t linear_id = static_cast<std::uint64_t>(layout_int);

  // Delegate to the three-parameter version with default auth_id=0 and packing_id=0
  return ConstructLayout(linear_id, 0, 0, layouts, mapping, break_on_failure);
}

  //
  // ConstructLayout() - Three-parameter version with separate layout_splitting_id, layout_auth_id, and layout_packing_id
  //
  std::vector<Status> Legal::ConstructLayout(uint64_t layout_splitting_id, uint64_t layout_packing_id, uint64_t layout_auth_id, layout::Layouts* layouts, Mapping mapping, bool break_on_failure)
  {
  (void)break_on_failure; // Suppress unused parameter warning

  // This function takes separate IDs for all three design spaces:
  // - layout_splitting_id: for SplittingSpace (intraline-to-interline splitting)
  // - layout_packing_id: for PackingSpace (interline-to-intraline packing)
  // - layout_auth_id: for AuthSpace (authblock factor variations)

  // Create a deep copy of the layout to ensure modifications don't affect the original
  CreateConcordantLayout(mapping);

#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "\n=== LAYOUT CONSTRUCTION START ===" << std::endl;
  std::cout << "Layout IDs: IntraLine=" << layout_splitting_id << ", Auth=" << layout_auth_id
            << ", Packing=" << layout_packing_id << std::endl;
  std::cout << "Initial original layout:" << std::endl;
  layout::PrintOverallLayoutConcise(layout_);
#endif

  /*
    Step 0: Sanity Checking
  */
  // If no variable factors, just return the original layout
  if (variable_authblock_factors_.empty() && splitting_options_per_level_.empty() && packing_options_per_level_.empty())
  {
    // Copy the current layout to the output parameter
    if (layouts != nullptr)
    {
      *layouts = layout_;
    }

    Status success_status;
    success_status.success = true;
    success_status.fail_reason = "";
    return {success_status};
  }

  // Validate layout_splitting_id range
  if (!splitting_options_per_level_.empty() && layout_splitting_id >= splitting_candidates)
  {
    Status error_status;
    error_status.success = false;
    error_status.fail_reason = "layout_splitting_id " + std::to_string(layout_splitting_id) + " exceeds SplittingSpace size " + std::to_string(splitting_candidates);
    return {error_status};
  }

  // Validate layout_packing_id range
  if (!packing_options_per_level_.empty() && layout_packing_id >= packing_candidates)
  {
    Status error_status;
    error_status.success = false;
    error_status.fail_reason = "layout_packing_id " + std::to_string(layout_packing_id) + " exceeds PackingSpace size " + std::to_string(packing_candidates);
    return {error_status};
  }

  // Validate layout_auth_id range
  if (!variable_authblock_factors_.empty() && layout_auth_id >= authblock_candidates)
  {
    Status error_status;
    error_status.success = false;
    error_status.fail_reason = "layout_auth_id " + std::to_string(layout_auth_id) + " exceeds AuthSpace size " + std::to_string(authblock_candidates);
    return {error_status};
  }

  /*
    Step 1: Decode the design space choices (Updated for both single and multi-rank splitting)
  */
  // Use the pre-computed splitting choices per level from CreateIntralineFactorSpace
  // This avoids division by zero and handles impossible levels correctly
  std::vector<bool> level_allows_no_splitting;
  
  for (size_t lvl = 0; lvl < splitting_choices_per_level_.size(); lvl++)
  {
    uint64_t single_rank_options = (lvl < splitting_options_per_level_.size()) ? splitting_options_per_level_[lvl].size() : 0;
    uint64_t multi_rank_options = (lvl < multi_rank_splitting_options_per_level_.size()) ? multi_rank_splitting_options_per_level_[lvl].size() : 0;
    uint64_t cross_dataspace_multi_rank_options = (lvl < cross_dataspace_multi_rank_splitting_options_per_level_.size()) ? cross_dataspace_multi_rank_splitting_options_per_level_[lvl].size() : 0;
    
    bool allows_no_splitting = false;
    if (lvl < level_requires_splitting_.size() && !level_requires_splitting_[lvl]) {
      allows_no_splitting = true;
    }
    level_allows_no_splitting.push_back(allows_no_splitting);
    
    // Check for impossible levels (require splitting but have no actual options)
    if (lvl < level_requires_splitting_.size() && level_requires_splitting_[lvl] && (single_rank_options + multi_rank_options + cross_dataspace_multi_rank_options) == 0)
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Level " + std::to_string(lvl) + " requires splitting but no splitting options are available";
      return {error_status};
    }
  }

  // Decode SplittingSpace choices using layout_splitting_id
  std::vector<int> splitting_choice_type_per_level(splitting_choices_per_level_.size(), -1); // -1: no splitting, 0: single-rank, 1: multi-rank, 2: cross-dataspace
  std::vector<uint32_t> splitting_option_index_per_level(splitting_choices_per_level_.size(), 0);
  std::uint64_t remaining_splitting_id = layout_splitting_id;

#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "Decoding splitting choices:" << std::endl;
  std::cout << "  splitting_choices_per_level_.size(): " << splitting_choices_per_level_.size() << std::endl;
  for (size_t i = 0; i < splitting_choices_per_level_.size(); i++) {
    std::cout << "  Level " << i << ": " << splitting_choices_per_level_[i] << " choices available" << std::endl;
  }
  std::cout << "  layout_splitting_id: " << layout_splitting_id << std::endl;
#endif

  for (size_t lvl = 0; lvl < splitting_choices_per_level_.size(); lvl++)
  {
    uint32_t choice_index = remaining_splitting_id % splitting_choices_per_level_[lvl];
    remaining_splitting_id /= splitting_choices_per_level_[lvl];

#ifdef DEBUG_CONSTRUCTION_LAYOUT
    std::cout << "  Level " << lvl << ": raw choice_index=" << choice_index 
              << ", level_allows_no_splitting=" << (level_allows_no_splitting[lvl] ? "true" : "false") << std::endl;
#endif

    // Handle conditional "no splitting" option
    if (level_allows_no_splitting[lvl] && choice_index == 0)
    {
      // No splitting (only available if level allows it)
      splitting_choice_type_per_level[lvl] = -1;
#ifdef DEBUG_CONSTRUCTION_LAYOUT
      std::cout << "    -> No splitting selected" << std::endl;
#endif
    }
    else
    {
      // Adjust choice_index based on whether "no splitting" option exists
      uint32_t adjusted_choice_index = choice_index;
      if (level_allows_no_splitting[lvl]) {
        adjusted_choice_index--; // Subtract 1 for "no splitting" option
      }
      
      uint64_t single_rank_options = (lvl < splitting_options_per_level_.size()) ? splitting_options_per_level_[lvl].size() : 0;
      uint64_t multi_rank_options = (lvl < multi_rank_splitting_options_per_level_.size()) ? multi_rank_splitting_options_per_level_[lvl].size() : 0;
      uint64_t cross_dataspace_multi_rank_options = (lvl < cross_dataspace_multi_rank_splitting_options_per_level_.size()) ? cross_dataspace_multi_rank_splitting_options_per_level_[lvl].size() : 0;

      // Check for impossible level: requires splitting but has no actual options
      if (!level_allows_no_splitting[lvl] && (single_rank_options + multi_rank_options + cross_dataspace_multi_rank_options) == 0)
      {
        Status error_status;
        error_status.success = false;
        error_status.fail_reason = "Level " + std::to_string(lvl) + " requires splitting but no splitting options are available (impossible configuration reached during layout construction)";
        return {error_status};
      }

#ifdef DEBUG_CONSTRUCTION_LAYOUT
      std::cout << "    adjusted_choice_index=" << adjusted_choice_index 
                << ", single_rank_options=" << single_rank_options 
                << ", multi_rank_options=" << multi_rank_options 
                << ", cross_dataspace_options=" << cross_dataspace_multi_rank_options << std::endl;
#endif

      if (adjusted_choice_index < single_rank_options)
      {
        // Single-rank splitting
        splitting_choice_type_per_level[lvl] = 0;
        splitting_option_index_per_level[lvl] = adjusted_choice_index;
#ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "    -> Single-rank splitting, option_index=" << adjusted_choice_index << std::endl;
#endif
      }
      else if (adjusted_choice_index < single_rank_options + multi_rank_options)
      {
        // Multi-rank splitting
        splitting_choice_type_per_level[lvl] = 1;
        splitting_option_index_per_level[lvl] = adjusted_choice_index - single_rank_options;
#ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "    -> Multi-rank splitting, option_index=" << (adjusted_choice_index - single_rank_options) << std::endl;
#endif
      }
      else if (adjusted_choice_index < single_rank_options + multi_rank_options + cross_dataspace_multi_rank_options)
      {
        // Cross-dataspace multi-rank splitting
        splitting_choice_type_per_level[lvl] = 2;
        splitting_option_index_per_level[lvl] = adjusted_choice_index - single_rank_options - multi_rank_options;
#ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "    -> Cross-dataspace splitting, option_index=" << (adjusted_choice_index - single_rank_options - multi_rank_options) << std::endl;
#endif
      }
      else
      {
        // Invalid choice index
#ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "    -> INVALID CHOICE INDEX!" << std::endl;
#endif
        Status error_status;
        error_status.success = false;
        error_status.fail_reason = "Invalid choice index " + std::to_string(adjusted_choice_index) + 
                                  " for level " + std::to_string(lvl) + 
                                  ". Available options: single=" + std::to_string(single_rank_options) + 
                                  ", multi=" + std::to_string(multi_rank_options) + 
                                  ", cross=" + std::to_string(cross_dataspace_multi_rank_options);
        return {error_status};
      }
    }
  }

  // Decode PackingSpace choices using layout_packing_id (interline-to-intraline packing)
  std::vector<int> packing_choice_type_per_level(packing_choices_per_level_.size(), -1); // -1: no packing, 0: single-rank, 1: multi-rank, 2: cross-dataspace
  std::vector<uint32_t> packing_option_index_per_level(packing_choices_per_level_.size(), 0);
  std::uint64_t remaining_packing_id = layout_packing_id;

  for (size_t level = 0; level < packing_choices_per_level_.size(); level++)
  {
    uint32_t choice_index = remaining_packing_id % packing_choices_per_level_[level];
    remaining_packing_id /= packing_choices_per_level_[level];

    // Choice 0 means "no packing" for this level
    if (choice_index == 0)
    {
      packing_choice_type_per_level[level] = -1;
    }
    else
    {
      // Adjust choice_index (subtract 1 for "no packing" option)
      uint32_t adjusted_choice_index = choice_index - 1;
      
      uint64_t single_rank_options = (level < packing_options_per_level_.size()) ? packing_options_per_level_[level].size() : 0;
      uint64_t multi_rank_options = (level < multi_rank_packing_options_per_level_.size()) ? multi_rank_packing_options_per_level_[level].size() : 0;
      uint64_t cross_dataspace_multi_rank_options = (level < cross_dataspace_multi_rank_packing_options_per_level_.size()) ? cross_dataspace_multi_rank_packing_options_per_level_[level].size() : 0;

      if (adjusted_choice_index < single_rank_options)
      {
        // Single-rank packing
        packing_choice_type_per_level[level] = 0;
        packing_option_index_per_level[level] = adjusted_choice_index;
      }
      else if (adjusted_choice_index < single_rank_options + multi_rank_options)
      {
        // Multi-rank packing
        packing_choice_type_per_level[level] = 1;
        packing_option_index_per_level[level] = adjusted_choice_index - single_rank_options;
      }
      else if (adjusted_choice_index < single_rank_options + multi_rank_options + cross_dataspace_multi_rank_options)
      {
        // Cross-dataspace multi-rank packing
        packing_choice_type_per_level[level] = 2;
        packing_option_index_per_level[level] = adjusted_choice_index - single_rank_options - multi_rank_options;
      }
      else
      {
        // Invalid choice index
        Status error_status;
        error_status.success = false;
        error_status.fail_reason = "Invalid packing choice index " + std::to_string(adjusted_choice_index) + 
                                  " for level " + std::to_string(level) + 
                                  ". Available options: single=" + std::to_string(single_rank_options) + 
                                  ", multi=" + std::to_string(multi_rank_options) + 
                                  ", cross=" + std::to_string(cross_dataspace_multi_rank_options);
        return {error_status};
      }
    }
  }

  std::vector<uint32_t> authblock_choices(variable_authblock_factors_.size());
  std::uint64_t remaining_auth_id = layout_auth_id;

  for (size_t i = 0; i < variable_authblock_factors_.size(); i++)
  {
    const auto& divisors = authblock_factor_ranges_[i];
    uint32_t divisor_index = remaining_auth_id % divisors.size();
    authblock_choices[i] = divisors[divisor_index];
    remaining_auth_id /= divisors.size();
  }

#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "=== LAYOUT CONSTRUCTION START ===" << std::endl;
  std::cout << "Layout IDs: IntraLine=" << layout_splitting_id << ", Auth=" << layout_auth_id << ", Packing=" << layout_packing_id << std::endl;
  std::cout << "Constructing layout with three separate IDs:" << std::endl;
  std::cout << "  SplittingSpace (layout_splitting_id): " << layout_splitting_id << ", choices per level: [";
  for (size_t i = 0; i < splitting_choice_type_per_level.size(); i++)
  {
    int choice_type = splitting_choice_type_per_level[i];
    if (choice_type == -1) {
      std::cout << "no-split";
    } else if (choice_type == 0) {
      std::cout << "single-rank:" << splitting_option_index_per_level[i];
    } else if (choice_type == 1) {
      std::cout << "multi-rank:" << splitting_option_index_per_level[i];
    } else if (choice_type == 2) {
      std::cout << "cross-dataspace:" << splitting_option_index_per_level[i];
    }
    if (i < splitting_choice_type_per_level.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << "  PackingSpace (layout_packing_id): " << layout_packing_id << ", per-level choices: [";
  for (size_t level = 0; level < packing_choice_type_per_level.size(); level++)
  {
    int choice_type = packing_choice_type_per_level[level];
    if (choice_type == -1) {
      std::cout << "no-pack";
    } else if (choice_type == 0) {
      std::cout << "single-rank:" << packing_option_index_per_level[level];
    } else if (choice_type == 1) {
      std::cout << "multi-rank:" << packing_option_index_per_level[level];
    } else if (choice_type == 2) {
      std::cout << "cross-dataspace:" << packing_option_index_per_level[level];
    }
    if (level < packing_choice_type_per_level.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << "  AuthSpace (layout_auth_id): " << layout_auth_id << ", choices: [";
  for (size_t i = 0; i < authblock_choices.size(); i++)
  {
    std::cout << authblock_choices[i];
    if (i < authblock_choices.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
#endif

    // Apply SplittingSpace choices (both single-rank and multi-rank splitting: intraline-to-interline)
#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "[SplittingSpace] Applying single-rank and multi-rank splitting..." << std::endl;
#endif

  for (size_t level = 0; level < splitting_choices_per_level_.size(); level++)
  {
    int choice_type = splitting_choice_type_per_level[level];
    uint32_t choice_index = splitting_option_index_per_level[level];

    // Validate that levels requiring splitting don't have "no splitting" choice
    if (level < level_requires_splitting_.size() && level_requires_splitting_[level] && choice_type == -1)
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Level " + std::to_string(level) + " requires splitting but 'no splitting' choice was selected";
      return {error_status};
    }

    // Choice -1 means "no splitting" for this level (only valid if splitting is optional)
    if (choice_type == -1)
    {
#ifdef DEBUG_CONSTRUCTION_LAYOUT
      std::cout << "[SplittingSpace] Storage Level " << level << ": No splitting applied (splitting is optional)" << std::endl;
#endif
      continue;
    }

    // Choice > 0 means apply the corresponding splitting option
    if (choice_type == 0) // Single-rank splitting
    {
      if (level >= splitting_options_per_level_.size() || choice_index >= splitting_options_per_level_[level].size())
      {
        Status error_status;
        error_status.success = false;
        error_status.fail_reason = "Invalid single-rank splitting choice " + std::to_string(choice_index) + " for level " + std::to_string(level);
        return {error_status};
      }

      // Get the selected splitting option
      const auto& splitting_option = splitting_options_per_level_[level][choice_index];

      unsigned ds_idx = splitting_option.dataspace;
      std::string rank = splitting_option.rank;
      uint32_t original_intraline_factor = splitting_option.original_intraline_factor;
      uint32_t splitting_factor = splitting_option.splitting_factor;

      // Validate indices
      if (level >= layout_.size())
      {
        Status error_status;
        error_status.success = false;
        error_status.fail_reason = "Invalid storage level " + std::to_string(level) + " in splitting option";
        return {error_status};
      }

      if (ds_idx >= layout_[level].intraline.size())
      {
        Status error_status;
        error_status.success = false;
        error_status.fail_reason = "Invalid data space index " + std::to_string(ds_idx) + " in splitting option";
        return {error_status};
      }

      // Apply the intraline-to-interline splitting
      auto& intraline_nest = layout_[level].intraline[ds_idx];
      auto& interline_nest = layout_[level].interline[ds_idx];

      // Check if rank exists in both nests
      auto intra_rank_it = std::find(intraline_nest.ranks.begin(), intraline_nest.ranks.end(), rank);
      auto inter_rank_it = std::find(interline_nest.ranks.begin(), interline_nest.ranks.end(), rank);

      if (intra_rank_it == intraline_nest.ranks.end() || inter_rank_it == interline_nest.ranks.end())
      {
        Status error_status;
        error_status.success = false;
        error_status.fail_reason = "Rank " + rank + " not found in intraline or interline nest for level " + std::to_string(level) + ", dataspace " + std::to_string(ds_idx);
        return {error_status};
      }

      // Split the factor: move splitting_factor from intraline to interline
      uint32_t new_intraline_factor = original_intraline_factor / splitting_factor;
      uint32_t current_interline_factor = (interline_nest.factors.find(rank) != interline_nest.factors.end()
                                          ? interline_nest.factors.at(rank) : 1);
      uint32_t new_interline_factor = current_interline_factor * splitting_factor;

#ifdef DEBUG_CONSTRUCTION_LAYOUT
      std::cout << "[SplittingSpace] Storage Level " << level << ", DataSpace " << ds_idx
                << ", Rank '" << rank << "': Splitting factor " << splitting_factor
                << " from intraline to interline (choice " << choice_index << ")" << std::endl;
      std::cout << "  - intraline factor: " << original_intraline_factor << " -> " << new_intraline_factor
                << " (divided by " << splitting_factor << ")" << std::endl;
      std::cout << "  - interline factor: " << current_interline_factor << " -> " << new_interline_factor
                << " (multiplied by " << splitting_factor << ")" << std::endl;
#endif

      intraline_nest.factors[rank] = new_intraline_factor;
      interline_nest.factors[rank] = new_interline_factor;
    }
    else if (choice_type == 1) // Multi-rank splitting
    {
      if (level >= multi_rank_splitting_options_per_level_.size() || choice_index >= multi_rank_splitting_options_per_level_[level].size())
      {
        Status error_status;
        error_status.success = false;
        error_status.fail_reason = "Invalid multi-rank splitting choice " + std::to_string(choice_index) + " for level " + std::to_string(level);
        return {error_status};
      }

      // Get the selected multi-rank splitting option
      const auto& multi_option = multi_rank_splitting_options_per_level_[level][choice_index];

      // Apply the multi-rank splitting
      for (const auto& rank : multi_option.ranks)
      {
        uint32_t original_intraline_factor = multi_option.original_intraline_factors.at(rank);
        uint32_t splitting_factor = multi_option.splitting_factors.at(rank);

        // Get references to both nests
        auto& intraline_nest = layout_[level].intraline[multi_option.dataspace];
        auto& interline_nest = layout_[level].interline[multi_option.dataspace];

        // Check if rank exists in both nests
        auto intra_rank_it = std::find(intraline_nest.ranks.begin(), intraline_nest.ranks.end(), rank);
        auto inter_rank_it = std::find(interline_nest.ranks.begin(), interline_nest.ranks.end(), rank);

        if (intra_rank_it == intraline_nest.ranks.end() || inter_rank_it == interline_nest.ranks.end())
        {
          Status error_status;
          error_status.success = false;
          error_status.fail_reason = "Rank " + rank + " not found in intraline or interline nest for level " + std::to_string(level) + ", dataspace " + std::to_string(multi_option.dataspace);
          return {error_status};
        }

#ifdef DEBUG_CONSTRUCTION_LAYOUT
        // Get the old factors for comparison BEFORE modifying them
        uint32_t old_intraline_factor = (intraline_nest.factors.find(rank) != intraline_nest.factors.end()
                                        ? intraline_nest.factors.at(rank) : 1);
        uint32_t old_interline_factor = (interline_nest.factors.find(rank) != interline_nest.factors.end()
                                        ? interline_nest.factors.at(rank) : 1);
#endif

        // Split the factor: move splitting_factor from intraline to interline
        uint32_t new_intraline_factor = original_intraline_factor / splitting_factor;
        uint32_t current_interline_factor = (interline_nest.factors.find(rank) != interline_nest.factors.end()
                                            ? interline_nest.factors.at(rank) : 1);
        uint32_t new_interline_factor = current_interline_factor * splitting_factor;

        // Apply the changes
        intraline_nest.factors[rank] = new_intraline_factor;
        interline_nest.factors[rank] = new_interline_factor;

#ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "[SplittingSpace] Storage Level " << level << ", DataSpace " << multi_option.dataspace
                  << ", Rank '" << rank << "': Multi-rank splitting factor " << splitting_factor
                  << " from intraline to interline" << std::endl;
        std::cout << "  - intraline factor: " << old_intraline_factor << " -> " << new_intraline_factor
                  << " (divided by " << splitting_factor << ")" << std::endl;
        std::cout << "  - interline factor: " << old_interline_factor << " -> " << new_interline_factor
                  << " (multiplied by " << splitting_factor << ")" << std::endl;
#endif
      }
    }
    else if (choice_type == 2) // Cross-dataspace multi-rank splitting
    {
      if (level >= cross_dataspace_multi_rank_splitting_options_per_level_.size() || choice_index >= cross_dataspace_multi_rank_splitting_options_per_level_[level].size())
      {
        Status error_status;
        error_status.success = false;
        error_status.fail_reason = "Invalid cross-dataspace multi-rank splitting choice " + std::to_string(choice_index) + " for level " + std::to_string(level);
        return {error_status};
      }

      // Get the selected cross-dataspace multi-rank splitting option
      const auto& cross_multi_option = cross_dataspace_multi_rank_splitting_options_per_level_[level][choice_index];

      // Apply the cross-dataspace multi-rank splitting
      for (const auto& unique_rank : cross_multi_option.ranks)
      {
        uint32_t splitting_factor = cross_multi_option.splitting_factors.at(unique_rank);
        unsigned dataspace_idx = cross_multi_option.rank_to_dataspace.at(unique_rank);
        
        // Extract the actual rank name (remove dataspace prefix)
        std::string actual_rank = unique_rank.substr(unique_rank.find('_') + 1);

#ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "[SplittingSpace] Processing cross-dataspace rank: " << unique_rank 
                  << " -> actual_rank: " << actual_rank << ", dataspace: " << dataspace_idx 
                  << ", splitting_factor: " << splitting_factor << std::endl;
#endif

        // Get references to both nests for the specific dataspace
        auto& intraline_nest = layout_[level].intraline[dataspace_idx];
        auto& interline_nest = layout_[level].interline[dataspace_idx];

        // Check if rank exists in both nests
        auto intra_rank_it = std::find(intraline_nest.ranks.begin(), intraline_nest.ranks.end(), actual_rank);
        auto inter_rank_it = std::find(interline_nest.ranks.begin(), interline_nest.ranks.end(), actual_rank);

        if (intra_rank_it == intraline_nest.ranks.end() || inter_rank_it == interline_nest.ranks.end())
        {
          Status error_status;
          error_status.success = false;
          error_status.fail_reason = "Rank " + actual_rank + " not found in intraline or interline nest for level " + std::to_string(level) + ", dataspace " + std::to_string(dataspace_idx) + ". Available intraline ranks: " + [&](){
            std::string ranks_str = "";
            for (const auto& r : intraline_nest.ranks) ranks_str += r + " ";
            return ranks_str;
          }();
          return {error_status};
        }

        // Get current factors from the layout (not from stored original values)
        uint32_t current_intraline_factor = (intraline_nest.factors.find(actual_rank) != intraline_nest.factors.end()
                                            ? intraline_nest.factors.at(actual_rank) : 1);
        uint32_t current_interline_factor = (interline_nest.factors.find(actual_rank) != interline_nest.factors.end()
                                            ? interline_nest.factors.at(actual_rank) : 1);

        // Validate that the splitting factor divides the current intraline factor
        if (current_intraline_factor % splitting_factor != 0)
        {
          Status error_status;
          error_status.success = false;
          error_status.fail_reason = "Cross-dataspace splitting factor " + std::to_string(splitting_factor) + 
                                    " does not divide current intraline factor " + std::to_string(current_intraline_factor) + 
                                    " for rank " + actual_rank + " at level " + std::to_string(level) + ", dataspace " + std::to_string(dataspace_idx);
          return {error_status};
        }

        // Split the factor: move splitting_factor from intraline to interline
        uint32_t new_intraline_factor = current_intraline_factor / splitting_factor;
        uint32_t new_interline_factor = current_interline_factor * splitting_factor;

#ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "[SplittingSpace] Storage Level " << level << ", DataSpace " << dataspace_idx
                  << ", Rank '" << actual_rank << "': Cross-dataspace multi-rank splitting factor " << splitting_factor
                  << " from intraline to interline" << std::endl;
        std::cout << "  - intraline factor: " << current_intraline_factor << " -> " << new_intraline_factor
                  << " (divided by " << splitting_factor << ")" << std::endl;
        std::cout << "  - interline factor: " << current_interline_factor << " -> " << new_interline_factor
                  << " (multiplied by " << splitting_factor << ")" << std::endl;
#endif

        // Apply the changes
        intraline_nest.factors[actual_rank] = new_intraline_factor;
        interline_nest.factors[actual_rank] = new_interline_factor;
      }
    }
  }

  // Apply PackingSpace choices (single-rank, multi-rank, and cross-dataspace packing)
#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "[PackingSpace] Applying single-rank, multi-rank, and cross-dataspace packing..." << std::endl;
#endif

  for (size_t level = 0; level < packing_choice_type_per_level.size(); level++)
  {
    int choice_type = packing_choice_type_per_level[level];
    uint32_t choice_index = packing_option_index_per_level[level];

    // Choice type -1 means "no packing" for this level
    if (choice_type == -1)
    {
#ifdef DEBUG_CONSTRUCTION_LAYOUT
      std::cout << "[PackingSpace] Storage Level " << level << ": No packing applied" << std::endl;
#endif
      continue;
    }

    // Apply the corresponding packing option based on choice type
    if (choice_type == 0) // Single-rank packing
    {
      if (level >= packing_options_per_level_.size() || choice_index >= packing_options_per_level_[level].size())
      {
        Status error_status;
        error_status.success = false;
        error_status.fail_reason = "Invalid single-rank packing choice " + std::to_string(choice_index) + " for level " + std::to_string(level);
        return {error_status};
      }

      // Get the selected packing option
      const auto& packing_option = packing_options_per_level_[level][choice_index];

    unsigned ds_idx = packing_option.dataspace;
    std::string rank = packing_option.rank;
    uint32_t original_interline_factor = packing_option.original_interline_factor;
    uint32_t packing_factor = packing_option.packing_factor;

    // Validate indices
    if (level >= layout_.size())
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Invalid storage level " + std::to_string(level) + " in packing option";
      return {error_status};
    }

    if (ds_idx >= layout_[level].intraline.size())
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Invalid data space index " + std::to_string(ds_idx) + " in packing option";
      return {error_status};
    }

    // Apply the interline-to-intraline packing
    auto& intraline_nest = layout_[level].intraline[ds_idx];
    auto& interline_nest = layout_[level].interline[ds_idx];

    // Check if rank exists in both nests
    auto intra_rank_it = std::find(intraline_nest.ranks.begin(), intraline_nest.ranks.end(), rank);
    auto inter_rank_it = std::find(interline_nest.ranks.begin(), interline_nest.ranks.end(), rank);

    if (intra_rank_it == intraline_nest.ranks.end() || inter_rank_it == interline_nest.ranks.end())
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Rank " + rank + " not found in intraline or interline nest for level " + std::to_string(level) + ", dataspace " + std::to_string(ds_idx);
      return {error_status};
    }

    // Move the packing factor from interline to intraline
    uint32_t current_intraline_factor = (intraline_nest.factors.find(rank) != intraline_nest.factors.end()
                                        ? intraline_nest.factors.at(rank) : 1);
    uint32_t new_interline_factor = original_interline_factor / packing_factor;
    uint32_t new_intraline_factor = current_intraline_factor * packing_factor;

#ifdef DEBUG_CONSTRUCTION_LAYOUT
    uint32_t current_interline_factor = (interline_nest.factors.find(rank) != interline_nest.factors.end()
    ? interline_nest.factors.at(rank) : 1);
    std::cout << "[PackingSpace] Storage Level " << level << ", DataSpace " << ds_idx
              << ", Rank '" << rank << "': Packing factor " << packing_factor
              << " from interline to intraline (choice " << choice_index << ")" << std::endl;
    std::cout << "  - interline factor: " << current_interline_factor << " -> " << new_interline_factor
              << " (divided by " << packing_factor << ")" << std::endl;
    std::cout << "  - intraline factor: " << current_intraline_factor << " -> " << new_intraline_factor
              << " (multiplied by " << packing_factor << ")" << std::endl;
#endif

      intraline_nest.factors[rank] = new_intraline_factor;
      interline_nest.factors[rank] = new_interline_factor;
    }
    else if (choice_type == 1) // Multi-rank packing
    {
      if (level >= multi_rank_packing_options_per_level_.size() || choice_index >= multi_rank_packing_options_per_level_[level].size())
      {
        Status error_status;
        error_status.success = false;
        error_status.fail_reason = "Invalid multi-rank packing choice " + std::to_string(choice_index) + " for level " + std::to_string(level);
        return {error_status};
      }

      // Get the selected multi-rank packing option
      const auto& multi_option = multi_rank_packing_options_per_level_[level][choice_index];

      // Apply the multi-rank packing
      for (const auto& rank : multi_option.ranks)
      {
        uint32_t packing_factor = multi_option.packing_factors.at(rank);
        unsigned dataspace_idx = multi_option.dataspace;

        // Get references to both nests for the specific dataspace
        auto& intraline_nest = layout_[level].intraline[dataspace_idx];
        auto& interline_nest = layout_[level].interline[dataspace_idx];

        // Check if rank exists in both nests
        auto intra_rank_it = std::find(intraline_nest.ranks.begin(), intraline_nest.ranks.end(), rank);
        auto inter_rank_it = std::find(interline_nest.ranks.begin(), interline_nest.ranks.end(), rank);

        if (intra_rank_it == intraline_nest.ranks.end() || inter_rank_it == interline_nest.ranks.end())
        {
          Status error_status;
          error_status.success = false;
          error_status.fail_reason = "Rank " + rank + " not found in intraline or interline nest for level " + std::to_string(level) + ", dataspace " + std::to_string(dataspace_idx);
          return {error_status};
        }

        // Get current factors from the layout
        uint32_t current_intraline_factor = (intraline_nest.factors.find(rank) != intraline_nest.factors.end()
                                            ? intraline_nest.factors.at(rank) : 1);
        uint32_t current_interline_factor = (interline_nest.factors.find(rank) != interline_nest.factors.end()
                                            ? interline_nest.factors.at(rank) : 1);

        // Validate that the packing factor divides the current interline factor
        if (current_interline_factor % packing_factor != 0)
        {
          Status error_status;
          error_status.success = false;
          error_status.fail_reason = "Multi-rank packing factor " + std::to_string(packing_factor) + 
                                    " does not divide current interline factor " + std::to_string(current_interline_factor) + 
                                    " for rank " + rank + " at level " + std::to_string(level) + ", dataspace " + std::to_string(dataspace_idx);
          return {error_status};
        }

        // Pack the factor: move packing_factor from interline to intraline
        uint32_t new_interline_factor = current_interline_factor / packing_factor;
        uint32_t new_intraline_factor = current_intraline_factor * packing_factor;

#ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "[PackingSpace] Storage Level " << level << ", DataSpace " << dataspace_idx
                  << ", Rank '" << rank << "': Multi-rank packing factor " << packing_factor
                  << " from interline to intraline" << std::endl;
        std::cout << "  - interline factor: " << current_interline_factor << " -> " << new_interline_factor
                  << " (divided by " << packing_factor << ")" << std::endl;
        std::cout << "  - intraline factor: " << current_intraline_factor << " -> " << new_intraline_factor
                  << " (multiplied by " << packing_factor << ")" << std::endl;
#endif

        // Apply the changes
        intraline_nest.factors[rank] = new_intraline_factor;
        interline_nest.factors[rank] = new_interline_factor;
      }
    }
    else if (choice_type == 2) // Cross-dataspace multi-rank packing
    {
      if (level >= cross_dataspace_multi_rank_packing_options_per_level_.size() || choice_index >= cross_dataspace_multi_rank_packing_options_per_level_[level].size())
      {
        Status error_status;
        error_status.success = false;
        error_status.fail_reason = "Invalid cross-dataspace multi-rank packing choice " + std::to_string(choice_index) + " for level " + std::to_string(level);
        return {error_status};
      }

      // Get the selected cross-dataspace multi-rank packing option
      const auto& cross_multi_option = cross_dataspace_multi_rank_packing_options_per_level_[level][choice_index];

      // Apply the cross-dataspace multi-rank packing
      for (const auto& unique_rank : cross_multi_option.ranks)
      {
        uint32_t packing_factor = cross_multi_option.packing_factors.at(unique_rank);
        unsigned dataspace_idx = cross_multi_option.rank_to_dataspace.at(unique_rank);
        
        // Extract the actual rank name (remove dataspace prefix)
        std::string actual_rank = unique_rank.substr(unique_rank.find('_') + 1);

#ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "[PackingSpace] Processing cross-dataspace rank: " << unique_rank 
                  << " -> actual_rank: " << actual_rank << ", dataspace: " << dataspace_idx 
                  << ", packing_factor: " << packing_factor << std::endl;
#endif

        // Get references to both nests for the specific dataspace
        auto& intraline_nest = layout_[level].intraline[dataspace_idx];
        auto& interline_nest = layout_[level].interline[dataspace_idx];

        // Check if rank exists in both nests
        auto intra_rank_it = std::find(intraline_nest.ranks.begin(), intraline_nest.ranks.end(), actual_rank);
        auto inter_rank_it = std::find(interline_nest.ranks.begin(), interline_nest.ranks.end(), actual_rank);

        if (intra_rank_it == intraline_nest.ranks.end() || inter_rank_it == interline_nest.ranks.end())
        {
          Status error_status;
          error_status.success = false;
          error_status.fail_reason = "Rank " + actual_rank + " not found in intraline or interline nest for level " + std::to_string(level) + ", dataspace " + std::to_string(dataspace_idx) + ". Available intraline ranks: " + [&](){
            std::string ranks_str = "";
            for (const auto& r : intraline_nest.ranks) ranks_str += r + " ";
            return ranks_str;
          }();
          return {error_status};
        }

        // Get current factors from the layout
        uint32_t current_intraline_factor = (intraline_nest.factors.find(actual_rank) != intraline_nest.factors.end()
                                            ? intraline_nest.factors.at(actual_rank) : 1);
        uint32_t current_interline_factor = (interline_nest.factors.find(actual_rank) != interline_nest.factors.end()
                                            ? interline_nest.factors.at(actual_rank) : 1);

        // Validate that the packing factor divides the current interline factor
        if (current_interline_factor % packing_factor != 0)
        {
          Status error_status;
          error_status.success = false;
          error_status.fail_reason = "Cross-dataspace packing factor " + std::to_string(packing_factor) + 
                                    " does not divide current interline factor " + std::to_string(current_interline_factor) + 
                                    " for rank " + actual_rank + " at level " + std::to_string(level) + ", dataspace " + std::to_string(dataspace_idx);
          return {error_status};
        }

        // Pack the factor: move packing_factor from interline to intraline
        uint32_t new_interline_factor = current_interline_factor / packing_factor;
        uint32_t new_intraline_factor = current_intraline_factor * packing_factor;

#ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "[PackingSpace] Storage Level " << level << ", DataSpace " << dataspace_idx
                  << ", Rank '" << actual_rank << "': Cross-dataspace multi-rank packing factor " << packing_factor
                  << " from interline to intraline" << std::endl;
        std::cout << "  - interline factor: " << current_interline_factor << " -> " << new_interline_factor
                  << " (divided by " << packing_factor << ")" << std::endl;
        std::cout << "  - intraline factor: " << current_intraline_factor << " -> " << new_intraline_factor
                  << " (multiplied by " << packing_factor << ")" << std::endl;
#endif

        // Apply the changes
        intraline_nest.factors[actual_rank] = new_intraline_factor;
        interline_nest.factors[actual_rank] = new_interline_factor;
      }
    }
  }


  // Apply AuthSpace factor choices (using layout_auth_id)
  for (size_t i = 0; i < variable_authblock_factors_.size(); i++)
  {
    auto& var_factor = variable_authblock_factors_[i];
    unsigned lvl = std::get<0>(var_factor);
    unsigned ds_idx = std::get<1>(var_factor);
    std::string rank = std::get<2>(var_factor);
    uint32_t chosen_factor = authblock_choices[i];

    // Apply the chosen factor to the authblock_lines nest
    auto& authblock_nest = layout_[lvl].authblock_lines[ds_idx];

    // Check if rank exists in the authblock nest
    auto rank_it = std::find(authblock_nest.ranks.begin(), authblock_nest.ranks.end(), rank);
    if (rank_it == authblock_nest.ranks.end())
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Rank " + rank + " not found in authblock_lines nest for level " + std::to_string(lvl) + ", dataspace " + std::to_string(ds_idx);
      return {error_status};
    }

    // Set the chosen factor value
    authblock_nest.factors[rank] = chosen_factor;

#ifdef DEBUG_CONSTRUCTION_LAYOUT
    // Get the old factor for comparison
    uint32_t old_authblock_factor = (authblock_nest.factors.find(rank) != authblock_nest.factors.end()
                                    ? authblock_nest.factors.at(rank) : 1);

    std::cout << "[AuthSpace] Storage Level " << lvl << ", DataSpace " << ds_idx
              << ", Rank '" << rank << "': authblock_lines factor "
              << old_authblock_factor << " -> " << chosen_factor << std::endl;
#endif
  }

  // Copy the modified layout to the output parameter
  if (layouts != nullptr)
  {
    *layouts = layout_;
  }

#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "\n=== LAYOUT CONSTRUCTION COMPLETE ===" << std::endl;
  std::cout << "Final modified layout:" << std::endl;
  layout::PrintOverallLayoutConcise(layout_);
#endif

  std::vector<std::uint64_t> intraline_size(num_storage_levels, 0);

  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++)
  {
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++)
    {
      // Check if this dataspace is bypassed at this storage level
      bool is_kept = mapping.datatype_bypass_nest.at(ds_idx).test(lvl);

      if (is_kept)
      {
        uint64_t intraline_per_ds = 1;
        auto intra_nest = layout_.at(lvl).intraline.at(ds_idx);
        for (const auto &r : intra_nest.ranks) // Analyze slowdown per rank
        {
        int factor = (intra_nest.factors.find(r) != intra_nest.factors.end() ? intra_nest.factors.at(r) : 1);
          intraline_per_ds *= factor;
        }
        intraline_size[lvl] += intraline_per_ds;
      }
    }
    if (intraline_size[lvl] > storage_level_line_capacity[lvl]){
      throw std::runtime_error("Intraline size " + std::to_string(intraline_size[lvl]) + " exceeds storage level line capacity " + std::to_string(storage_level_line_capacity[lvl]) + " at level " + std::to_string(lvl));
    }
  }

  // Return success status
#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "=== LAYOUT CONSTRUCTION COMPLETE ===" << std::endl;
  std::cout << "Final modified layout:" << std::endl;
  layout::PrintOverallLayout(layout_);
#endif
  
  Status success_status;
  success_status.success = true;
  success_status.fail_reason = "";

  return {success_status};
}


//
// CreateConcordantLayout() - Step 1: Create layout from mapping
//
void Legal::CreateConcordantLayout(const Mapping& mapping)
{
#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "Step 1: Create Concordant Layout..." << std::endl;
  std::cout << "Total number of storage levels: " << mapping.loop_nest.storage_tiling_boundaries.size() << std::endl;
  std::cout << "Total number of layout levels: " << layout_.size() << std::endl;
  assert(mapping.loop_nest.storage_tiling_boundaries.size() == layout_.size());
  std::cout << "Total number of data spaces: " << layout_.at(0).intraline.size() << std::endl;
#endif

  // Build a initialized map that assigns 1 to every dimension ID present in dim_order.
  std::map<std::uint32_t, std::uint32_t> initial_dimid_to_loopend;
  for (char dim_char : layout_.at(0).dim_order)
  {
    // Convert the char stored in dim_order to a std::string so it can be used
    // as a key into the dimensionToDimID map.
    std::string dim_name(1, dim_char);

    // Look up the dimension ID associated with this name.
    auto dim_id_itr = layout_.at(0).dimensionToDimID.find(dim_name);
    if (dim_id_itr == layout_.at(0).dimensionToDimID.end())
    {
      std::cerr << "ERROR: dimension name " << dim_name << " not found in dimensionToDimID map." << std::endl;
      throw std::runtime_error("Invalid dimension name in dim_order");
    }

    initial_dimid_to_loopend[dim_id_itr->second] = 1;
  }

  /*
      Step 1: Collect the interline nested loop and intraline nested loop.
  */
  num_storage_levels = mapping.loop_nest.storage_tiling_boundaries.size();
  num_data_spaces = layout_.at(0).intraline.size();
  unsigned num_loops = mapping.loop_nest.loops.size();
  unsigned inv_storage_level = num_storage_levels;

  // Each storage level vector element starts as a copy of the prototype map.
  std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_interline_dimid_to_loopend(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);
  std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_intraline_dimid_to_loopend(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);
  std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_overall_dimval(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);

  for (unsigned loop_level = num_loops-1; loop_level != static_cast<unsigned>(-1); loop_level--)
  {
    if (inv_storage_level > 0 &&
        mapping.loop_nest.storage_tiling_boundaries.at(inv_storage_level-1) == loop_level)
    {
      inv_storage_level--;
    }

    if (loop::IsSpatial(mapping.loop_nest.loops.at(loop_level).spacetime_dimension))
    {
      storage_level_intraline_dimid_to_loopend[inv_storage_level][mapping.loop_nest.loops.at(loop_level).dimension] = mapping.loop_nest.loops.at(loop_level).end;
    }else{
      storage_level_interline_dimid_to_loopend.at(inv_storage_level)[mapping.loop_nest.loops.at(loop_level).dimension] = mapping.loop_nest.loops.at(loop_level).end;
    }
  }

  for(unsigned lvl=0; lvl < storage_level_intraline_dimid_to_loopend.size(); lvl++){
    for (unsigned i = 0; i < num_data_spaces; i++){ // iterate over all data
      for (const auto& kv : storage_level_interline_dimid_to_loopend[lvl])
      {
        storage_level_overall_dimval[lvl][kv.first] = storage_level_intraline_dimid_to_loopend[lvl][kv.first] * storage_level_interline_dimid_to_loopend[lvl][kv.first];
      }
    }
  }

  // Calculate cumulative product from end to first index
  cumulatively_intraline_dimval.resize(storage_level_intraline_dimid_to_loopend.size());

  // Initialize all levels with the initial map
  for (unsigned lvl = 0; lvl < cumulatively_intraline_dimval.size(); lvl++)
  {
    cumulatively_intraline_dimval[lvl] = initial_dimid_to_loopend;
  }

  // Initialize the last level (no multiplication needed)
  if (!storage_level_intraline_dimid_to_loopend.empty())
  {
    cumulatively_intraline_dimval[0] = storage_level_intraline_dimid_to_loopend[0];

    // Calculate cumulative product from second-to-last level backwards to first level
    for (int lvl = 1; lvl < static_cast<int>(storage_level_intraline_dimid_to_loopend.size()); lvl++)
    {
      for (const auto& kv : storage_level_intraline_dimid_to_loopend[lvl])
      {
        std::uint32_t dim_id = kv.first;
        std::uint32_t current_value = kv.second;

        // Multiply current level value with cumulative product from next level
        if (cumulatively_intraline_dimval[lvl - 1].find(dim_id) != cumulatively_intraline_dimval[lvl - 1].end())
        {
          cumulatively_intraline_dimval[lvl][dim_id] = current_value * cumulatively_intraline_dimval[lvl - 1][dim_id];
        }
        else
        {
          cumulatively_intraline_dimval[lvl][dim_id] = current_value;
        }
      }
    }
  }

  // Calculate cumulative product from end to first index
  cumulatively_product_dimval.resize(storage_level_overall_dimval.size());

  // Initialize all levels with the initial map
  for (unsigned lvl = 0; lvl < cumulatively_product_dimval.size(); lvl++)
  {
    cumulatively_product_dimval[lvl] = initial_dimid_to_loopend;
  }

  // Initialize the last level (no multiplication needed)
  if (!storage_level_overall_dimval.empty())
  {
    cumulatively_product_dimval[0] = storage_level_overall_dimval[0];

    // Calculate cumulative product from second-to-last level backwards to first level
    for (int lvl = 1; lvl < static_cast<int>(storage_level_overall_dimval.size()); lvl++)
    {
      for (const auto& kv : storage_level_overall_dimval[lvl])
      {
        std::uint32_t dim_id = kv.first;
        std::uint32_t current_value = kv.second;

        // Multiply current level value with cumulative product from next level
        if (cumulatively_product_dimval[lvl - 1].find(dim_id) != cumulatively_product_dimval[lvl - 1].end())
        {
          cumulatively_product_dimval[lvl][dim_id] = current_value * cumulatively_product_dimval[lvl - 1][dim_id];
        }
        else
        {
          cumulatively_product_dimval[lvl][dim_id] = current_value;
        }
      }
    }
  }

  /*
      Step 2: Print out the collapsed interline nested loop and intraline nested loop.
  */
#ifdef DEBUG_CONCORDANT_LAYOUT
  std::cout << "storage_level_interline_dimid_to_loopend:" << std::endl;
  for (unsigned lvl = 0; lvl < storage_level_interline_dimid_to_loopend.size(); lvl++) // iterate over all storage levels
  {
    std::cout << "storage level=" << lvl << std::endl;
    for (const auto& kv : storage_level_interline_dimid_to_loopend[lvl])
    {
      std::cout << layout_.at(0).dim_order[kv.first] << ":" << kv.second << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "storage_level_intraline_dimid_to_loopend:" << std::endl;
  for (unsigned lvl = 0; lvl < storage_level_intraline_dimid_to_loopend.size(); lvl++) // iterate over all storage levels
  {
    std::cout << "storage level=" << lvl << std::endl;
    for (const auto& kv : storage_level_intraline_dimid_to_loopend[lvl])
    {
      std::cout << layout_.at(0).dim_order[kv.first] << ":" << kv.second << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "storage_level_overall_dimval:" << std::endl;
  for (unsigned lvl = 0; lvl < storage_level_overall_dimval.size(); lvl++) // iterate over all storage levels
  {
    std::cout << "storage level=" << lvl << std::endl;
    for (const auto& kv : storage_level_overall_dimval[lvl])
    {
      std::cout << layout_.at(0).dim_order[kv.first] << ":" << kv.second << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "cumulatively_product_dimval:" << std::endl;
  for (unsigned lvl = 0; lvl < cumulatively_product_dimval.size(); lvl++) // iterate over all storage levels
  {
    std::cout << "storage level=" << lvl << std::endl;
    for (const auto& kv : cumulatively_product_dimval[lvl])
    {
      std::cout << layout_.at(0).dim_order[kv.first] << ":" << kv.second << " ";
    }
    std::cout << std::endl;
  }
#endif

  /*
      Step 3: Assign collapsed nested loop to the layout.
  */
  for(unsigned lvl=0; lvl < cumulatively_intraline_dimval.size(); lvl++){
    for (unsigned i = 0; i < num_data_spaces; i++){ // iterate over all data spaces
      for(auto & rank: layout_.at(lvl).intraline.at(i).ranks){ // iterate over all ranks of the data space
        const auto& dim_ids = layout_.at(lvl).rankToFactorizedDimensionID.at(rank);
        uint32_t total_intraline = 0;
        uint32_t total_rank_size = 0;
        const auto& coefficient = layout_.at(lvl).rankToCoefficientValue[rank];
        for (unsigned idx=0; idx < dim_ids.size(); idx++){
          auto dim_intraline_value = cumulatively_intraline_dimval[lvl][dim_ids[idx]];
          auto dim_total_value = cumulatively_product_dimval[lvl][dim_ids[idx]];
          if (dim_ids.size() > 1){
            if (dim_intraline_value == 1){
              if (idx < dim_ids.size()-1){
                total_intraline += dim_intraline_value;
              }
            }
            else{
              if (idx < dim_ids.size()-1){
                total_intraline += dim_intraline_value*coefficient[idx];
              }
              else{
                total_intraline += dim_intraline_value*coefficient[idx] - 1;
              }
            }

            if (dim_total_value == 1){
              if (idx < dim_ids.size()-1){
                total_rank_size += dim_total_value;
              }
            }
            else{
              if (idx < dim_ids.size()-1){
                total_rank_size += dim_total_value*coefficient[idx];
              }
              else{
                total_rank_size += dim_total_value*coefficient[idx] - 1;
              }
            }
          }
          else{
            total_intraline += dim_intraline_value;
            total_rank_size += dim_total_value;
          }
        }
        auto total_interline = (total_rank_size + total_intraline - 1) / total_intraline;

        layout_.at(lvl).intraline.at(i).factors.at(rank) = total_intraline;
        layout_.at(lvl).interline.at(i).factors.at(rank) = total_interline;
#ifdef DEBUG_CONCORDANT_LAYOUT
        std::cout << "level=" << lvl << " dataspace=" << i << " rank=" << rank << " intraline = " << total_intraline << " interline = " << total_interline << std::endl;
#endif
      }
    }
  }

#ifdef DEBUG_CONCORDANT_LAYOUT
  std::cout << "layout_after_concordant_layout:" << std::endl;
  layout::PrintOverallLayout(layout_);
#endif
}

//
// CreateIntralineFactorSpace() - Step 3: Generate all possible intraline factor combinations (SplittingSpace and PackingSpace)
//
void Legal::CreateIntralineFactorSpace(model::Engine::Specs arch_specs, const Mapping& mapping)
{
  (void) arch_specs; // Suppress unused parameter warning

  std::cout << "Step 2: Creating SplittingSpace candidates from intraline factors..." << std::endl;

  // Clear previous design spaces
  splitting_options_per_level_.clear();
  splitting_choices_per_level_.clear();
  packing_options_per_level_.clear();
  packing_choices_per_level_.clear();
  multi_rank_splitting_options_per_level_.clear();
  level_requires_splitting_.clear();

  // Phase 1: Get Memory Line size for all storage levels (What Layout Provide Per Cycle)
  std::vector<std::vector<std::uint64_t>> intraline_size_per_ds(num_storage_levels, std::vector<std::uint64_t>(num_data_spaces, 0));
  std::vector<std::uint64_t> intraline_size_per_lvl(num_storage_levels, 0);

  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++)
  {
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++)
    {
      // Check if this dataspace is bypassed at this storage level
      bool is_kept = mapping.datatype_bypass_nest.at(ds_idx).test(lvl);

      if (is_kept)
      {
        uint64_t intraline_per_ds = 1;
        auto intra_nest = layout_.at(lvl).intraline.at(ds_idx);
        for (const auto &r : intra_nest.ranks) // Analyze slowdown per rank
        {
        int factor = (intra_nest.factors.find(r) != intra_nest.factors.end() ? intra_nest.factors.at(r) : 1);
          intraline_per_ds *= factor;
        }
        intraline_size_per_ds[lvl][ds_idx] = intraline_per_ds;
        intraline_size_per_lvl[lvl] += intraline_per_ds;
      }
    }
  }


  // Phase 2: Check if the line capacity is sufficient for the intraline size
  
  // First, determine which levels require splitting (intraline_size > line_capacity)
  level_requires_splitting_.resize(num_storage_levels, false);
  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++){
    if(storage_level_line_capacity[lvl] < intraline_size_per_lvl[lvl]){
      level_requires_splitting_[lvl] = true;
      std::cout << "  Level " << lvl << ": requires splitting (intraline_size=" << intraline_size_per_lvl[lvl] 
                << " > line_capacity=" << storage_level_line_capacity[lvl] << ")" << std::endl;
    } else {
      level_requires_splitting_[lvl] = false;
      std::cout << "  Level " << lvl << ": splitting optional (intraline_size=" << intraline_size_per_lvl[lvl] 
                << " <= line_capacity=" << storage_level_line_capacity[lvl] << ")" << std::endl;
    }
  }
  
  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++){
    if(storage_level_line_capacity[lvl] < intraline_size_per_lvl[lvl]){
      // The product of all factors of intraline for a dataspace is too big to fit in the line capacity,
      // so need to reduce the factors of intraline by converting some factors into interline.

      std::cout << "  Level " << lvl << ": intraline_size (" << intraline_size_per_lvl[lvl]
                << ") exceeds line capacity (" << storage_level_line_capacity[lvl]
                << "). Generating design space for factor conversions..." << std::endl;

      // Calculate maximum packing factor that can be applied
      uint32_t max_splitting_factor = static_cast<uint32_t>((static_cast<float>(intraline_size_per_lvl[lvl]) + static_cast<float>(storage_level_line_capacity[lvl]) - 1) / static_cast<float>(storage_level_line_capacity[lvl]));
      if (max_splitting_factor > 1){
        std::cout << "    Maximum splitting factor: " << max_splitting_factor << std::endl;

        // For each dataspace, analyze packing possibilities
        for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
          auto& intraline_nest = layout_.at(lvl).intraline.at(ds_idx);
          std::cout << "    DataSpace " << ds_idx << ":" << std::endl;
          
          // First, collect all possible splitting factors for each rank (for multi-rank combinations)
          std::map<std::string, std::vector<uint32_t>> all_candidate_factors_per_rank;
          for (const auto& rank : intraline_nest.ranks) {
            uint32_t current_intraline_factor = (intraline_nest.factors.find(rank) != intraline_nest.factors.end()
                                                ? intraline_nest.factors.at(rank) : 1);
            if (current_intraline_factor > 1) {
              std::vector<uint32_t> divisors = FindDivisors(current_intraline_factor);
              std::vector<uint32_t> valid_factors;
              
              // Store all divisors > 1 as candidate factors for multi-rank combinations
              for (uint32_t divisor : divisors) {
                if (divisor > 1) {
                  valid_factors.push_back(divisor);
                }
              }
              
              if (!valid_factors.empty()) {
                all_candidate_factors_per_rank[rank] = valid_factors;
                std::cout << "      Rank " << rank << ": candidate factors [";
                for (size_t i = 0; i < valid_factors.size(); i++) {
                  std::cout << valid_factors[i];
                  if (i < valid_factors.size() - 1) std::cout << ", ";
                }
                std::cout << "] (for multi-rank combinations)" << std::endl;
              }
            }
          }

          // Single-rank splitting: find factors that can be moved from intraline to interline (individual fit requirement)
          for (const auto& rank : intraline_nest.ranks) {
            uint32_t current_intraline_factor = (intraline_nest.factors.find(rank) != intraline_nest.factors.end()
                                                ? intraline_nest.factors.at(rank) : 1);

            if (current_intraline_factor > 1) {
              std::vector<uint32_t> divisors = FindDivisors(current_intraline_factor);
              std::vector<uint32_t> valid_splitting_factors;

               // Test each divisor (excluding 1) to see if it can be split
               for (uint32_t divisor : divisors) {
                 if (divisor > 1) { // Skip 1 as it means no splitting
                   // Calculate new total intraline_size if this divisor is moved from intraline to interline
                   uint64_t new_total_intraline_size = 0;
                   for (unsigned ds_idx_inner = 0; ds_idx_inner < num_data_spaces; ds_idx_inner++){
                     if (ds_idx_inner == ds_idx) {
                       new_total_intraline_size += intraline_size_per_ds[lvl][ds_idx_inner] / divisor;
                     } else {
                       new_total_intraline_size += intraline_size_per_ds[lvl][ds_idx_inner];
                     }
                   }

                  if (new_total_intraline_size <= storage_level_line_capacity[lvl]) {
                    // This divisor fits - add to design space
                    valid_splitting_factors.push_back(divisor);
                    std::cout << "      Rank " << rank << ": splitting factor " << divisor
                              << " gives total_intraline_size=" << new_total_intraline_size
                              << " (fits in capacity - single-rank option)" << std::endl;
                  } else {
                    std::cout << "      Rank " << rank << ": splitting factor " << divisor
                              << " gives total_intraline_size=" << new_total_intraline_size
                              << " (exceeds capacity - not viable for single-rank)" << std::endl;
                    break; // No need to test larger factors for this rank
                  }
                }
              }

              // Store in member variables for later use by ConstructLayout (new per-level structure)
              if (!valid_splitting_factors.empty()) {
                // Ensure we have space for this storage level
                while (splitting_options_per_level_.size() <= lvl) {
                  splitting_options_per_level_.push_back(std::vector<SplittingOption>());
                }

                // Add each packing factor as a separate option for this level
                for (uint32_t splitting_factor : valid_splitting_factors) {
                  SplittingOption option;
                  option.dataspace = ds_idx;
                  option.rank = rank;
                  option.original_intraline_factor = current_intraline_factor;
                  option.splitting_factor = splitting_factor;
                  splitting_options_per_level_[lvl].push_back(option);
                }

                std::cout << "      Stored splitting options for Level " << lvl
                        << ", DataSpace " << ds_idx << ", Rank " << rank
                        << ", intraline_factor: " << current_intraline_factor
                        << ", splitting_factors: [";
                for (size_t i = 0; i < valid_splitting_factors.size(); i++) {
                  std::cout << valid_splitting_factors[i];
                  if (i < valid_splitting_factors.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
              }
            }
          }

          // Multi-rank splitting: find combinations of ranks that together can reduce intraline size to fit
          // Use all candidate factors (including those that don't fit individually)
          std::cout << "      Multi-rank splitting analysis:" << std::endl;
          std::vector<std::vector<std::string>> rank_combinations = GenerateRankCombinations(intraline_nest.ranks);
          
          // Ensure we have space for multi-rank options at this storage level
          while (multi_rank_splitting_options_per_level_.size() <= lvl) {
            multi_rank_splitting_options_per_level_.push_back(std::vector<MultiRankSplittingOption>());
          }

          for (const auto& rank_combo : rank_combinations) {
            // Check if all ranks in combination have candidate factors
            bool all_ranks_have_factors = true;
            for (const auto& rank : rank_combo) {
              if (all_candidate_factors_per_rank.find(rank) == all_candidate_factors_per_rank.end()) {
                all_ranks_have_factors = false;
                break;
              }
            }
            
            if (!all_ranks_have_factors) continue;
            
            MultiRankSplittingOption multi_option;
            if (TestMultiRankSplittingWithCandidates(lvl, ds_idx, rank_combo, all_candidate_factors_per_rank, 
                                                   intraline_size_per_ds, storage_level_line_capacity[lvl], multi_option)) {
              multi_rank_splitting_options_per_level_[lvl].push_back(multi_option);
              
              std::cout << "        Valid multi-rank option: Ranks [";
              for (size_t i = 0; i < rank_combo.size(); i++) {
                std::cout << rank_combo[i];
                if (i < rank_combo.size() - 1) std::cout << ", ";
              }
              std::cout << "], total_reduction=" << multi_option.total_reduction << std::endl;
              
              // Print individual splitting factors
              for (const auto& rank : rank_combo) {
                std::cout << "          " << rank << ": " << multi_option.original_intraline_factors.at(rank)
                          << " -> " << (multi_option.original_intraline_factors.at(rank) / multi_option.splitting_factors.at(rank))
                          << " (split by " << multi_option.splitting_factors.at(rank) << ")" << std::endl;
              }
            }
          }
        }

        // Cross-dataspace multi-rank splitting analysis
        // Generate combinations of ranks across ALL dataspaces at this level
        std::cout << "      Cross-dataspace multi-rank splitting analysis:" << std::endl;
        
        // First, collect all ranks and their candidate factors from all dataspaces
        std::map<std::string, std::vector<uint32_t>> all_cross_dataspace_candidate_factors;
        std::map<std::string, std::pair<unsigned, uint32_t>> rank_to_dataspace_and_original_factor; // rank -> {dataspace_idx, original_factor}
        
        for (unsigned ds_idx_cross = 0; ds_idx_cross < num_data_spaces; ds_idx_cross++) {
          auto& intraline_nest_cross = layout_.at(lvl).intraline.at(ds_idx_cross);
          
          for (const auto& rank : intraline_nest_cross.ranks) {
            uint32_t current_intraline_factor = (intraline_nest_cross.factors.find(rank) != intraline_nest_cross.factors.end()
                                                ? intraline_nest_cross.factors.at(rank) : 1);
            
            if (current_intraline_factor > 1) {
              std::vector<uint32_t> divisors = FindDivisors(current_intraline_factor);
              std::vector<uint32_t> valid_factors;
              
              for (uint32_t divisor : divisors) {
                if (divisor > 1) {
                  valid_factors.push_back(divisor);
                }
              }
              
              if (!valid_factors.empty()) {
                // Create unique rank identifier with dataspace prefix
                std::string unique_rank = "DS" + std::to_string(ds_idx_cross) + "_" + rank;
                all_cross_dataspace_candidate_factors[unique_rank] = valid_factors;
                rank_to_dataspace_and_original_factor[unique_rank] = {ds_idx_cross, current_intraline_factor};
              }
            }
          }
        }
        
        // Generate combinations of ranks across dataspaces (limit to reasonable size)
        std::vector<std::string> all_cross_dataspace_ranks;
        for (const auto& entry : all_cross_dataspace_candidate_factors) {
          all_cross_dataspace_ranks.push_back(entry.first);
        }
        
        if (all_cross_dataspace_ranks.size() >= 2) {
          std::vector<std::vector<std::string>> cross_dataspace_rank_combinations = GenerateRankCombinations(all_cross_dataspace_ranks, 4); // Limit to max 4 ranks
          
          // Ensure we have space for cross-dataspace multi-rank options at this storage level
          while (cross_dataspace_multi_rank_splitting_options_per_level_.size() <= lvl) {
            cross_dataspace_multi_rank_splitting_options_per_level_.push_back(std::vector<CrossDataspaceMultiRankSplittingOption>());
          }
          
          for (const auto& rank_combo : cross_dataspace_rank_combinations) {
            // Skip combinations that only involve ranks from a single dataspace (already handled above)
            std::set<unsigned> involved_dataspaces;
            for (const auto& unique_rank : rank_combo) {
              involved_dataspaces.insert(rank_to_dataspace_and_original_factor.at(unique_rank).first);
            }
            
            if (involved_dataspaces.size() < 2) continue; // Skip single-dataspace combinations
            
            // Check if all ranks in combination have candidate factors
            bool all_ranks_have_factors = true;
            for (const auto& unique_rank : rank_combo) {
              if (all_cross_dataspace_candidate_factors.find(unique_rank) == all_cross_dataspace_candidate_factors.end()) {
                all_ranks_have_factors = false;
                break;
              }
            }
            
            if (!all_ranks_have_factors) continue;
            
            CrossDataspaceMultiRankSplittingOption cross_multi_option;
            if (TestCrossDataspaceMultiRankSplittingWithCandidates(lvl, rank_combo, all_cross_dataspace_candidate_factors, 
                                                                 rank_to_dataspace_and_original_factor, intraline_size_per_ds, 
                                                                 storage_level_line_capacity[lvl], cross_multi_option)) {
              cross_dataspace_multi_rank_splitting_options_per_level_[lvl].push_back(cross_multi_option);
              
              std::cout << "        Valid cross-dataspace multi-rank option: Ranks [";
              for (size_t i = 0; i < rank_combo.size(); i++) {
                std::cout << rank_combo[i];
                if (i < rank_combo.size() - 1) std::cout << ", ";
              }
              std::cout << "], total_reduction=" << cross_multi_option.total_reduction << std::endl;
              
              // Print individual splitting factors with dataspace info
              for (const auto& unique_rank : rank_combo) {
                std::cout << "          " << unique_rank << ": " << cross_multi_option.original_intraline_factors.at(unique_rank)
                          << " -> " << (cross_multi_option.original_intraline_factors.at(unique_rank) / cross_multi_option.splitting_factors.at(unique_rank))
                          << " (split by " << cross_multi_option.splitting_factors.at(unique_rank) << ")" << std::endl;
              }
            }
          }
        } else {
          std::cout << "        Insufficient cross-dataspace ranks for combinations" << std::endl;
        }
      }
      else{
        std::cout << "  Level " << lvl << ": splitting factor = 1, no splitting is needed." << std::endl;
      }
      // Calculate maximum splitting factor that can be applied
    }
    else if (storage_level_line_capacity[lvl] > intraline_size_per_lvl[lvl]){
      // Intraline has free space to hold more data, could convert some factors of interline into intraline,
      // this creates the overall design spaces

      std::cout << "  Level " << lvl << ": intraline_size (" << intraline_size_per_lvl[lvl]
                << ") has " << (storage_level_line_capacity[lvl] - intraline_size_per_lvl[lvl])
                << " free capacity. Generating design space for data packing..." << std::endl;

      // Calculate maximum packing factor that can be applied
      uint32_t max_packing_factor = static_cast<uint32_t>(static_cast<float>(storage_level_line_capacity[lvl]) / static_cast<float>(intraline_size_per_lvl[lvl]));
      if (max_packing_factor > 1){
        std::cout << "    Maximum packing factor: " << max_packing_factor << std::endl;

        // For each dataspace, analyze packing possibilities
        for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
          auto& inter_nest = layout_.at(lvl).interline.at(ds_idx);

          std::cout << "    DataSpace " << ds_idx << ":" << std::endl;

          // Single-rank packing: find factors that can be moved from interline to intraline
          for (const auto& rank : inter_nest.ranks) {
            uint32_t current_interline_factor = (inter_nest.factors.find(rank) != inter_nest.factors.end()
                                                ? inter_nest.factors.at(rank) : 1);

            if (current_interline_factor > 1) {
              std::vector<uint32_t> divisors = FindDivisors(current_interline_factor);
              std::vector<uint32_t> valid_packing_factors;

              // Test each divisor (excluding 1) to see if it can be packed
              for (uint32_t divisor : divisors) {
                if (divisor > 1) { // Skip 1 as it means no packing
                  // Calculate new intraline_size if this divisor is moved to intraline
                  std::vector<uint64_t> new_intraline_size_per_ds(num_data_spaces, 0);
                  for (unsigned ds_idx_inner = 0; ds_idx_inner < num_data_spaces; ds_idx_inner++){
                    if (ds_idx_inner == ds_idx) {
                      new_intraline_size_per_ds[ds_idx_inner] = intraline_size_per_ds[lvl][ds_idx_inner] * divisor;
                    } else {
                      new_intraline_size_per_ds[ds_idx_inner] = intraline_size_per_ds[lvl][ds_idx_inner];
                    }
                  }
                  uint64_t new_intraline_size = std::accumulate(new_intraline_size_per_ds.begin(), new_intraline_size_per_ds.end(), 0);

                  if (new_intraline_size <= storage_level_line_capacity[lvl]) {
                    // This divisor fits - add to design space
                    valid_packing_factors.push_back(divisor);
                    std::cout << "      Rank " << rank << ": packing factor " << divisor
                              << " gives intraline_size=" << new_intraline_size
                              << " (fits in capacity)" << std::endl;
                  } else {
                    std::cout << "      Rank " << rank << ": packing factor " << divisor
                              << " gives intraline_size=" << new_intraline_size
                              << " (exceeds capacity)" << std::endl;
                    break; // No need to test larger factors for this rank
                  }
                }
              }

              // Store in member variables for later use by ConstructLayout (new per-level structure)
              if (!valid_packing_factors.empty()) {
                // Ensure we have space for this storage level
                while (packing_options_per_level_.size() <= lvl) {
                  packing_options_per_level_.push_back(std::vector<PackingOption>());
                }

                // Add each packing factor as a separate option for this level
                for (uint32_t packing_factor : valid_packing_factors) {
                  PackingOption option;
                  option.dataspace = ds_idx;
                  option.rank = rank;
                  option.original_interline_factor = current_interline_factor;
                  option.packing_factor = packing_factor;
                  packing_options_per_level_[lvl].push_back(option);
                }

                std::cout << "      Stored packing options for Level " << lvl
                        << ", DataSpace " << ds_idx << ", Rank " << rank
                        << ", interline_factor: " << current_interline_factor
                        << ", packing_factors: [";
                for (size_t i = 0; i < valid_packing_factors.size(); i++) {
                  std::cout << valid_packing_factors[i];
                  if (i < valid_packing_factors.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
              }
            }
          }

          // Multi-rank packing analysis within each dataspace
          std::cout << "    DataSpace " << ds_idx << " Multi-rank packing analysis:" << std::endl;
          
          // Collect all ranks and their candidate factors from this dataspace
          std::map<std::string, std::vector<uint32_t>> all_candidate_factors_per_rank;
          for (const auto& rank : inter_nest.ranks) {
            uint32_t current_interline_factor = (inter_nest.factors.find(rank) != inter_nest.factors.end()
                                                ? inter_nest.factors.at(rank) : 1);
            
            if (current_interline_factor > 1) {
              std::vector<uint32_t> divisors = FindDivisors(current_interline_factor);
              std::vector<uint32_t> valid_factors;
              
              for (uint32_t divisor : divisors) {
                if (divisor > 1) {
                  valid_factors.push_back(divisor);
                }
              }
              
              if (!valid_factors.empty()) {
                all_candidate_factors_per_rank[rank] = valid_factors;
              }
            }
          }
          
          if (all_candidate_factors_per_rank.size() >= 2) {
            std::vector<std::string> all_ranks;
            for (const auto& entry : all_candidate_factors_per_rank) {
              all_ranks.push_back(entry.first);
            }
            
            std::vector<std::vector<std::string>> rank_combinations = GenerateRankCombinations(all_ranks, 4); // Limit to max 4 ranks
            
            // Ensure we have space for multi-rank options at this storage level
            while (multi_rank_packing_options_per_level_.size() <= lvl) {
              multi_rank_packing_options_per_level_.push_back(std::vector<MultiRankPackingOption>());
            }
            
            for (const auto& rank_combo : rank_combinations) {
              MultiRankPackingOption multi_option;
              if (TestMultiRankPackingWithCandidates(lvl, ds_idx, rank_combo, all_candidate_factors_per_rank, 
                                                   intraline_size_per_ds, storage_level_line_capacity[lvl], multi_option)) {
                multi_rank_packing_options_per_level_[lvl].push_back(multi_option);
                
                std::cout << "      Valid multi-rank packing option: Ranks [";
                for (size_t i = 0; i < rank_combo.size(); i++) {
                  std::cout << rank_combo[i];
                  if (i < rank_combo.size() - 1) std::cout << ", ";
                }
                std::cout << "], total_packing=" << multi_option.total_packing << std::endl;
                
                // Print individual packing factors
                for (const auto& rank : rank_combo) {
                  std::cout << "        " << rank << ": " << multi_option.original_interline_factors.at(rank)
                            << " -> " << (multi_option.original_interline_factors.at(rank) / multi_option.packing_factors.at(rank))
                            << " (pack by " << multi_option.packing_factors.at(rank) << ")" << std::endl;
                }
              }
            }
          } else {
            std::cout << "      Insufficient ranks for multi-rank combinations in this dataspace" << std::endl;
          }
        }

        // Cross-dataspace multi-rank packing analysis
        // Generate combinations of ranks across ALL dataspaces at this level
        std::cout << "    Cross-dataspace multi-rank packing analysis:" << std::endl;
        
        // First, collect all ranks and their candidate factors from all dataspaces
        std::map<std::string, std::vector<uint32_t>> all_cross_dataspace_candidate_factors;
        std::map<std::string, std::pair<unsigned, uint32_t>> rank_to_dataspace_and_original_factor; // rank -> {dataspace_idx, original_factor}
        
        for (unsigned ds_idx_cross = 0; ds_idx_cross < num_data_spaces; ds_idx_cross++) {
          auto& interline_nest_cross = layout_.at(lvl).interline.at(ds_idx_cross);
          
          for (const auto& rank : interline_nest_cross.ranks) {
            uint32_t current_interline_factor = (interline_nest_cross.factors.find(rank) != interline_nest_cross.factors.end()
                                                ? interline_nest_cross.factors.at(rank) : 1);
            
            if (current_interline_factor > 1) {
              std::vector<uint32_t> divisors = FindDivisors(current_interline_factor);
              std::vector<uint32_t> valid_factors;
              
              for (uint32_t divisor : divisors) {
                if (divisor > 1) {
                  valid_factors.push_back(divisor);
                }
              }
              
              if (!valid_factors.empty()) {
                // Create unique rank identifier with dataspace prefix
                std::string unique_rank = "DS" + std::to_string(ds_idx_cross) + "_" + rank;
                all_cross_dataspace_candidate_factors[unique_rank] = valid_factors;
                rank_to_dataspace_and_original_factor[unique_rank] = {ds_idx_cross, current_interline_factor};
              }
            }
          }
        }
        
        // Generate combinations of ranks across dataspaces (limit to reasonable size)
        std::vector<std::string> all_cross_dataspace_ranks;
        for (const auto& entry : all_cross_dataspace_candidate_factors) {
          all_cross_dataspace_ranks.push_back(entry.first);
        }
        
        if (all_cross_dataspace_ranks.size() >= 2) {
          std::vector<std::vector<std::string>> cross_dataspace_rank_combinations = GenerateRankCombinations(all_cross_dataspace_ranks, 4); // Limit to max 4 ranks
          
          // Ensure we have space for cross-dataspace multi-rank packing options at this storage level
          while (cross_dataspace_multi_rank_packing_options_per_level_.size() <= lvl) {
            cross_dataspace_multi_rank_packing_options_per_level_.push_back(std::vector<CrossDataspaceMultiRankPackingOption>());
          }
          
          for (const auto& rank_combo : cross_dataspace_rank_combinations) {
            // Skip combinations that only involve ranks from a single dataspace (already handled above)
            std::set<unsigned> involved_dataspaces;
            for (const auto& unique_rank : rank_combo) {
              involved_dataspaces.insert(rank_to_dataspace_and_original_factor.at(unique_rank).first);
            }
            
            if (involved_dataspaces.size() < 2) continue; // Skip single-dataspace combinations
            
            CrossDataspaceMultiRankPackingOption cross_multi_option;
            if (TestCrossDataspaceMultiRankPackingWithCandidates(lvl, rank_combo, all_cross_dataspace_candidate_factors, 
                                                               rank_to_dataspace_and_original_factor, intraline_size_per_ds, 
                                                               storage_level_line_capacity[lvl], cross_multi_option)) {
              cross_dataspace_multi_rank_packing_options_per_level_[lvl].push_back(cross_multi_option);
              
              std::cout << "      Valid cross-dataspace multi-rank packing option: Ranks [";
              for (size_t i = 0; i < rank_combo.size(); i++) {
                std::cout << rank_combo[i];
                if (i < rank_combo.size() - 1) std::cout << ", ";
              }
              std::cout << "], total_packing=" << cross_multi_option.total_packing << std::endl;
              
              // Print individual packing factors with dataspace info
              for (const auto& unique_rank : rank_combo) {
                std::cout << "        " << unique_rank << ": " << cross_multi_option.original_interline_factors.at(unique_rank)
                          << " -> " << (cross_multi_option.original_interline_factors.at(unique_rank) / cross_multi_option.packing_factors.at(unique_rank))
                          << " (pack by " << cross_multi_option.packing_factors.at(unique_rank) << ")" << std::endl;
              }
            }
          }
        } else {
          std::cout << "      Insufficient cross-dataspace ranks for combinations" << std::endl;
        }
      }
      else{
        std::cout << "  Level " << lvl << ": packing factor = 1, no packing is needed." << std::endl;
      }
      // Setup choices per level (including "no packing" option)
      // This is done after all packing options for this level have been collected
    }
    // Do nothing if the line capacity is equal to the intraline size
  }

  // Setup choices per level and print summary of splitting design space
  if (!splitting_options_per_level_.empty() || !multi_rank_splitting_options_per_level_.empty() || !cross_dataspace_multi_rank_splitting_options_per_level_.empty()) {
    // Set up the number of choices for each level (including "no splitting" option)
    size_t max_levels = std::max({splitting_options_per_level_.size(), multi_rank_splitting_options_per_level_.size(), cross_dataspace_multi_rank_splitting_options_per_level_.size()});
    splitting_choices_per_level_.resize(max_levels);

    for (size_t level = 0; level < max_levels; level++) {
      uint64_t single_rank_options = (level < splitting_options_per_level_.size()) ? splitting_options_per_level_[level].size() : 0;
      uint64_t multi_rank_options = (level < multi_rank_splitting_options_per_level_.size()) ? multi_rank_splitting_options_per_level_[level].size() : 0;
      uint64_t cross_dataspace_multi_rank_options = (level < cross_dataspace_multi_rank_splitting_options_per_level_.size()) ? cross_dataspace_multi_rank_splitting_options_per_level_[level].size() : 0;
      
      // Only add +1 for "no splitting" option if splitting is optional for this level
      uint64_t no_splitting_options = 0;
      if (level < level_requires_splitting_.size() && !level_requires_splitting_[level]) {
        no_splitting_options = 1; // "no splitting" is allowed
        std::cout << "    Level " << level << ": 'no splitting' option available (splitting is optional)" << std::endl;
      } else {
        std::cout << "    Level " << level << ": 'no splitting' option removed (splitting required)" << std::endl;
      }
      
      uint64_t total_choices = no_splitting_options + single_rank_options + multi_rank_options + cross_dataspace_multi_rank_options;
      
      // For impossible levels (require splitting but have no options), store 1 to avoid division by zero in ConstructLayout
      // The actual impossibility will be caught during layout construction
      if (total_choices == 0) {
        splitting_choices_per_level_[level] = 1;
        std::cout << "    Level " << level << ": total choices = " << no_splitting_options << " (no-split) + " 
                  << single_rank_options << " (single-rank) + " << multi_rank_options << " (multi-rank) + "
                  << cross_dataspace_multi_rank_options << " (cross-dataspace) = " 
                  << total_choices << " → adjusted to 1 (impossible level)" << std::endl;
      } else {
        splitting_choices_per_level_[level] = total_choices;
        std::cout << "    Level " << level << ": total choices = " << no_splitting_options << " (no-split) + " 
                  << single_rank_options << " (single-rank) + " << multi_rank_options << " (multi-rank) + "
                  << cross_dataspace_multi_rank_options << " (cross-dataspace) = " 
                  << splitting_choices_per_level_[level] << std::endl;
      }
    }

    std::cout << "  Total splitting options across all levels: ";
    uint64_t total_single_rank_options = 0;
    uint64_t total_multi_rank_options = 0;
    uint64_t total_cross_dataspace_multi_rank_options = 0;
    for (const auto& level_options : splitting_options_per_level_) {
      total_single_rank_options += level_options.size();
    }
    for (const auto& level_options : multi_rank_splitting_options_per_level_) {
      total_multi_rank_options += level_options.size();
    }
    for (const auto& level_options : cross_dataspace_multi_rank_splitting_options_per_level_) {
      total_cross_dataspace_multi_rank_options += level_options.size();
    }
    
    // Calculate total number of splitting candidates correctly (product across levels)
    uint64_t level_based_splitting_candidates = 1;
    for (const auto& choices : splitting_choices_per_level_) {
      level_based_splitting_candidates *= choices;
    }

    splitting_candidates = level_based_splitting_candidates;

    std::cout << total_single_rank_options << " single-rank + " << total_multi_rank_options << " multi-rank + " 
              << total_cross_dataspace_multi_rank_options << " cross-dataspace = " 
              << (total_single_rank_options + total_multi_rank_options + total_cross_dataspace_multi_rank_options) << std::endl;
    std::cout << "  Intraline (splitting) layout candidates: " << splitting_candidates << std::endl;
  }

  // Setup choices per level and print summary of packing design space
  packing_candidates = 0;
  if (!packing_options_per_level_.empty() || !multi_rank_packing_options_per_level_.empty() || !cross_dataspace_multi_rank_packing_options_per_level_.empty()) {
    // Set up the number of choices for each level (including "no packing" option)
    size_t max_levels = std::max({packing_options_per_level_.size(), multi_rank_packing_options_per_level_.size(), cross_dataspace_multi_rank_packing_options_per_level_.size()});
    packing_choices_per_level_.resize(max_levels);
    
    for (size_t level = 0; level < max_levels; level++) {
      uint64_t single_rank_options = (level < packing_options_per_level_.size()) ? packing_options_per_level_[level].size() : 0;
      uint64_t multi_rank_options = (level < multi_rank_packing_options_per_level_.size()) ? multi_rank_packing_options_per_level_[level].size() : 0;
      uint64_t cross_dataspace_multi_rank_options = (level < cross_dataspace_multi_rank_packing_options_per_level_.size()) ? cross_dataspace_multi_rank_packing_options_per_level_[level].size() : 0;
      
      // +1 for "no packing" option (choice 0)
      packing_choices_per_level_[level] = 1 + single_rank_options + multi_rank_options + cross_dataspace_multi_rank_options;
      
      std::cout << "    Level " << level << ": total packing choices = 1 (no-pack) + " 
                << single_rank_options << " (single-rank) + " << multi_rank_options << " (multi-rank) + "
                << cross_dataspace_multi_rank_options << " (cross-dataspace) = " 
                << packing_choices_per_level_[level] << std::endl;
    }

    // Calculate total number of packing candidates
    packing_candidates = 1;
    for (const auto& choices : packing_choices_per_level_) {
      packing_candidates *= choices;
    }

    std::cout << "  Total packing options across all levels: ";
    uint64_t total_single_rank_options = 0;
    uint64_t total_multi_rank_options = 0;
    uint64_t total_cross_dataspace_multi_rank_options = 0;
    
    for (const auto& level_options : packing_options_per_level_) {
      total_single_rank_options += level_options.size();
    }
    for (const auto& level_options : multi_rank_packing_options_per_level_) {
      total_multi_rank_options += level_options.size();
    }
    for (const auto& level_options : cross_dataspace_multi_rank_packing_options_per_level_) {
      total_cross_dataspace_multi_rank_options += level_options.size();
    }
    
    std::cout << total_single_rank_options << " single-rank + " << total_multi_rank_options << " multi-rank + " 
              << total_cross_dataspace_multi_rank_options << " cross-dataspace = " 
              << (total_single_rank_options + total_multi_rank_options + total_cross_dataspace_multi_rank_options) << std::endl;
    std::cout << "  Packing layout candidates: " << packing_candidates << std::endl;
  }
}

//
// CreateAuthSpace() - Step 3: Generate all possible authblock_lines factor combinations
//
void Legal::CreateAuthSpace(model::Engine::Specs arch_specs)
{
  (void) arch_specs; // Suppress unused parameter warning

  std::cout << "Step 3: Creating layout candidate space from authblock_lines factors..." << std::endl;

  num_storage_levels = layout_.size();
  num_data_spaces = layout_.at(0).intraline.size();

  // Identify storage levels with non-empty authblock_lines and collect variable factors
  variable_authblock_factors_.clear();

  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++)
  {
    bool has_non_empty_authblock = false;

    // Check if authblock_lines vector has enough elements first
    if (layout_.at(lvl).authblock_lines.size() >= num_data_spaces)
    {
      for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++)
      {
        if (!layout_.at(lvl).authblock_lines.at(ds_idx).factors.empty())
        {
          has_non_empty_authblock = true;
          break;
        }
      }
    }

    if (has_non_empty_authblock)
    {
      std::cout << "  Level " << lvl << " has non-empty authblock_lines, will generate candidates" << std::endl;

      // Collect variable authblock_lines factors for this level
      for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++)
      {
        const auto& authblock_nest = layout_.at(lvl).authblock_lines.at(ds_idx);

        // Check if this nest has non-empty factors
        if (!authblock_nest.factors.empty())
        {
          for (const auto& rank : authblock_nest.ranks)
          {
            // Calculate max_factor as product of ratios between consecutive levels
            uint32_t max_factor = 1;

            // Get dimension IDs for this rank
            auto dims = layout_.at(lvl).rankToFactorizedDimensionID.at(rank);

            // Calculate product of ratios for all dimensions of this rank
            // Only proceed if we have at least 2 levels for comparison
            if (lvl >= 2)
            {
              for (uint32_t dim_id : dims)
              {
                auto cumulative_it_lvl = cumulatively_product_dimval[lvl-1].find(dim_id);
                auto cumulative_it_lvl_minus_1 = cumulatively_product_dimval[lvl-2].find(dim_id);

                if (cumulative_it_lvl != cumulatively_product_dimval[lvl-1].end() &&
                    cumulative_it_lvl_minus_1 != cumulatively_product_dimval[lvl-2].end() &&
                    cumulative_it_lvl_minus_1->second != 0)
                {
                  uint32_t ratio = cumulative_it_lvl->second / cumulative_it_lvl_minus_1->second;
                  max_factor *= ratio;
                }
                else
                {
                  std::cout << "Warning: dimension ID " << dim_id << " not found or zero division in cumulatively_product_dimval for level " << lvl-1 << " or " << (lvl-2) << std::endl;
                }
              }
            }

            std::cout << " lvl=" << lvl << " ds_idx=" << ds_idx << " rank=" << rank << " dims=[";
            for (size_t i = 0; i < dims.size(); i++)
            {
              std::cout << dims[i];
              if (i < dims.size() - 1) std::cout << ",";
            }
            std::cout << "] max_factor(product of ratios cumulatively_product_dimval[" << lvl-1 << "]/cumulatively_product_dimval[" << (lvl-2) << "])=" << max_factor << std::endl;

            // Only add if max_factor > 1 (there are variations possible)
            if (max_factor > 1)
            {
              variable_authblock_factors_.push_back(std::make_tuple(lvl, ds_idx, rank, max_factor));

              // Show the divisors that will be used
              std::vector<uint32_t> divisors = FindDivisors(max_factor);
              std::cout << "  Variable factor: Level " << lvl
                       << ", DataSpace " << ds_idx
                       << ", Rank " << rank
                       << ", Dimensions: [";
              for (size_t i = 0; i < dims.size(); i++)
              {
                std::cout << dims[i];
                if (i < dims.size() - 1) std::cout << ",";
              }
              std::cout << "], max_factor: " << max_factor << ", divisors: [";
              for (size_t i = 0; i < divisors.size(); i++)
              {
                std::cout << divisors[i];
                if (i < divisors.size() - 1) std::cout << ",";
              }
              std::cout << "]" << std::endl;
            }
          }
        }
      }
    }
    else
    {
      std::cout << "  Level " << lvl << " has empty authblock_lines, skipping candidate generation" << std::endl;
    }
  }

  // Calculate total number of combinations from authblock factors
  authblock_candidates = 1;
  if (!variable_authblock_factors_.empty())
  {
    authblock_factor_ranges_.clear();
    for (const auto& var_factor : variable_authblock_factors_)
    {
      uint32_t max_factor = std::get<3>(var_factor);
      std::vector<uint32_t> divisors = FindDivisors(max_factor);
      authblock_factor_ranges_.push_back(divisors); // Store all divisors of max_factor
      authblock_candidates *= divisors.size();
    }
    std::cout << "  Authblock_lines layout candidates: " << authblock_candidates << std::endl;
  }
  else
  {
    std::cout << "  No variable authblock_lines factors found." << std::endl;
  }

  // Total layout candidates is the sum of all three design spaces
  num_layout_candidates = authblock_candidates + splitting_candidates + packing_candidates;
  std::cout << "  Total combined layout candidates: " << num_layout_candidates << std::endl;
  std::cout << "  Variable factors count: " << variable_authblock_factors_.size() << std::endl;
  std::cout << "  Note: Only using divisors of max_factor for each variable factor" << std::endl;
  std::cout << "  ✓ Layout candidate space created successfully" << std::endl;
}

} // namespace layoutspace
