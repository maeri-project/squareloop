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

  // Generate all possible combinations all ranks (limited by ranks.size())
  for (size_t combo_size = 1; combo_size <= std::min(max_combo_size, ranks.size()); combo_size++)
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

#ifdef TESMULTIRANKCOMBINATION
      std::cout << "        Testing combination with accumulated_reduction=" << accumulated_reduction
                << ", new_dataspace_intraline_size=" << new_dataspace_intraline_size << std::endl;
#endif

      if (new_dataspace_intraline_size <= line_capacity) {
        // This combination works - store it in the option
        option.total_reduction = accumulated_reduction;
        for (size_t i = 0; i < rank_combination.size(); i++) {
          option.splitting_factors[rank_combination[i]] = current_factors[i];
        }
#ifdef TESMULTIRANKCOMBINATION
        std::cout << "          Success: Found valid combination with total reduction " << accumulated_reduction << std::endl;
        std::cout << "          Option details:" << std::endl;
        std::cout << "            Dataspace: " << option.dataspace << std::endl;
        std::cout << "            Ranks: [";
        for (size_t i = 0; i < option.ranks.size(); i++) {
          std::cout << option.ranks[i];
          if (i < option.ranks.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "            Original factors: {";
        for (const auto& pair : option.original_intraline_factors) {
          std::cout << pair.first << ": " << pair.second;
          if (pair != *option.original_intraline_factors.rbegin()) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
        std::cout << "            Splitting factors: {";
        for (const auto& pair : option.splitting_factors) {
          std::cout << pair.first << ": " << pair.second;
          if (pair != *option.splitting_factors.rbegin()) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
#endif
        return true;
      }
      return false;
    }

    // Try each candidate factor for the current rank
    const auto& rank = rank_combination[rank_idx];
    const auto& factors = candidate_factors_list[rank_idx];
    uint32_t original_factor = option.original_intraline_factors.at(rank);

    std::cout << "        Testing rank " << rank << " (index " << rank_idx << ") with original factor " << original_factor << std::endl;

    for (uint32_t factor : factors) {
      // Check if this factor is valid (i.e., divides the original factor)
      if (original_factor % factor == 0) {
        std::cout << "          Trying factor " << factor << " for rank " << rank << std::endl;
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

  // Variables to track the best (maximal) packing
  uint64_t best_packing = 0;
  std::vector<uint32_t> best_factors(rank_combination.size(), 0);

  // Generate all combinations of candidate factors using nested loops
  std::function<void(size_t, std::vector<uint32_t>&, uint64_t)> try_combinations =
    [&](size_t rank_idx, std::vector<uint32_t>& current_factors, uint64_t accumulated_packing) {

    if (rank_idx == rank_combination.size()) {
      // All ranks have been assigned factors, test if this combination works
      // Calculate the new intraline size for the split dataspace
      uint64_t new_dataspace_intraline_size = intraline_size_per_ds[lvl][ds_idx] * accumulated_packing;

      if (new_dataspace_intraline_size <= line_capacity) {
        // This combination works - check if it's the best so far
        if (accumulated_packing > best_packing) {
          best_packing = accumulated_packing;
          best_factors = current_factors;
        }
      }
      return;
    }

    // Try each candidate factor for the current rank
    const auto& rank = rank_combination[rank_idx];
    const auto& factors = candidate_factors_list[rank_idx];
    uint32_t original_factor = option.original_interline_factors.at(rank);

    for (auto factor_it = factors.rbegin(); factor_it != factors.rend(); ++factor_it) {
      // Check if this factor is valid (i.e., divides the original factor)
      if (original_factor % *factor_it == 0) {
        current_factors[rank_idx] = *factor_it;
        uint64_t new_accumulated_packing = accumulated_packing * *factor_it;

        // Recursive call for next rank
        try_combinations(rank_idx + 1, current_factors, new_accumulated_packing);
      }
    }
  };

  // Start the recursive combination testing
  std::vector<uint32_t> current_factors(rank_combination.size());
  try_combinations(0, current_factors, 1);

  // If a valid combination was found, update option and return true
  if (best_packing > 0) {
    option.total_packing = best_packing;
    option.packing_factors.clear();
    for (size_t i = 0; i < rank_combination.size(); i++) {
      option.packing_factors[rank_combination[i]] = best_factors[i];
    }
    return true;
  }
  return false;
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
  storage_level_bypass_factor.resize(num_storage_levels, std::vector<bool>(num_data_spaces, false));

  for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++){
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
      storage_level_bypass_factor[storage_level][ds_idx] = mapping.datatype_bypass_nest.at(ds_idx).test(storage_level);
    }
  }

  // Initialize the storage level capacity vectors
  storage_level_total_capacity.resize(num_storage_levels, 0);
  storage_level_line_capacity.resize(num_storage_levels, 0);

  // Iterate through each storage level to extract capacity information and bypass information.
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
    std::cout << "    Storage storage level " << storage_level
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

  // Decode SplittingSpace choices using layout_splitting_id (intraline-to-interline splitting)
  std::vector<std::vector<std::uint64_t>> splitting_choice_per_lvl_per_ds(num_storage_levels, std::vector<std::uint64_t>(num_data_spaces, 0));
  for (unsigned lvl = num_storage_levels-1; lvl-- > 0;) {
    for (unsigned ds_idx = num_data_spaces-1; ds_idx-- > 0;) {
      splitting_choice_per_lvl_per_ds[lvl][ds_idx] = layout_splitting_id % splitting_candidates_per_lvl_per_ds[lvl][ds_idx];
      layout_splitting_id = layout_splitting_id - splitting_choice_per_lvl_per_ds[lvl][ds_idx];
      layout_splitting_id = layout_splitting_id / splitting_candidates_per_lvl_per_ds[lvl][ds_idx];
    }
  }

  // Print flattened splitting choices
#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "Splitting choices:" << std::endl;
  std::cout << "Level | DataSpace | Choice" << std::endl;
  std::cout << "------|-----------|--------" << std::endl;
  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++) {
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++) {
      std::cout << std::setw(6) << lvl << " | " 
                << std::setw(9) << ds_idx << " | "
                << std::setw(6) << splitting_choice_per_lvl_per_ds[lvl][ds_idx] << std::endl;
    }
  }
  std::cout << std::endl;
#endif

  // Decode PackingSpace choices using layout_packing_id (interline-to-intraline packing)
  std::vector<std::vector<std::uint64_t>> packing_choice_per_lvl_per_ds(num_storage_levels, std::vector<std::uint64_t>(num_data_spaces, 0));
  for (unsigned lvl = num_storage_levels-1; lvl-- > 0;) {
    for (unsigned ds_idx = num_data_spaces-1; ds_idx-- > 0;) {
      packing_choice_per_lvl_per_ds[lvl][ds_idx] = layout_packing_id % packing_candidates_per_lvl_per_ds[lvl][ds_idx];
      layout_packing_id = layout_packing_id - packing_choice_per_lvl_per_ds[lvl][ds_idx];
      layout_packing_id = layout_packing_id / packing_candidates_per_lvl_per_ds[lvl][ds_idx];
    }
  }

  // Print flattened packing choices
#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "Packing choices:" << std::endl;
  std::cout << "Level | DataSpace | Choice" << std::endl;
  std::cout << "------|-----------|--------" << std::endl;
  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++) {
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++) {
      std::cout << std::setw(6) << lvl << " | " 
                << std::setw(9) << ds_idx << " | "
                << std::setw(6) << packing_choice_per_lvl_per_ds[lvl][ds_idx] << std::endl;
    }
  }
  std::cout << std::endl;
#endif

  // Decode AuthBlockSpace choices using layout_auth_id (authblock factor variations)
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
  std::cout << "Layout IDs: Splitting=" << layout_splitting_id  << ", Packing=" << layout_packing_id << ", Auth=" << layout_auth_id << std::endl;
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
  std::cout << "[SplittingSpace] Applying multi-rank splitting for all dataspaces..." << std::endl;
#endif

  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++) {
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++) {
      uint64_t choice = splitting_choice_per_lvl_per_ds[lvl][ds_idx];
      if (!(choice < multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx].size())){
#ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "Note: Do not need to split for storage level " << lvl << ", dataspace " << ds_idx << " because data fits in line capacity." << std::endl;
#endif
        continue;
      }
      const auto& multi_rank_option = multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx][choice];

      for (const auto& unique_rank : multi_rank_option.ranks)
      {
        uint32_t splitting_factor = multi_rank_option.splitting_factors.at(unique_rank);
        {
          std::stringstream ss;
          ss << "Layout vector size (" << layout_.size() << ") is smaller than or equal to current level index (" << lvl << ")";
          assert(layout_.size() > lvl && ss.str().c_str());
        }
        {
          std::stringstream ss;
          ss << "Intraline vector size (" << layout_[lvl].intraline.size() << ") is smaller than or equal to current dataspace index (" << ds_idx << ")";
          assert(layout_[lvl].intraline.size() > ds_idx && ss.str().c_str());
        }
        {
          std::stringstream ss;
          ss << "Interline vector size (" << layout_[lvl].interline.size() << ") is smaller than or equal to current dataspace index (" << ds_idx << ")";
          assert(layout_[lvl].interline.size() > ds_idx && ss.str().c_str());
        }
        // Get references to both nests for the specific dataspace
        auto& intraline_nest = layout_[lvl].intraline[ds_idx];
        auto& interline_nest = layout_[lvl].interline[ds_idx];
        
        if (intraline_nest.factors.find(unique_rank) == intraline_nest.factors.end() || interline_nest.factors.find(unique_rank) == interline_nest.factors.end())
        {
          Status error_status;
          error_status.success = false;
          error_status.fail_reason = "Rank " + unique_rank + " not found in intraline or interline nest for level " + std::to_string(lvl) + ", dataspace " + std::to_string(ds_idx);
          return {error_status};
        }

        // Get current factors from the layout (not from stored original values)
        uint32_t current_intraline_factor = (intraline_nest.factors.find(unique_rank) != intraline_nest.factors.end()
                                            ? intraline_nest.factors.at(unique_rank) : 1);
        uint32_t current_interline_factor = (interline_nest.factors.find(unique_rank) != interline_nest.factors.end()
                                            ? interline_nest.factors.at(unique_rank) : 1);

        // Validate that the splitting factor divides the current intraline factor
        if (current_intraline_factor % splitting_factor != 0)
        {
          Status error_status;
          error_status.success = false;
          error_status.fail_reason = "Multi-rank splitting factor " + std::to_string(splitting_factor) +
                                    " does not divide current intraline factor " + std::to_string(current_intraline_factor) +
                                    " for rank " + unique_rank + " at level " + std::to_string(lvl) + ", dataspace " + std::to_string(ds_idx);
          return {error_status};
        }

        // Split the factor: move splitting_factor from intraline to interline
        uint32_t new_intraline_factor = current_intraline_factor / splitting_factor;
        uint32_t new_interline_factor = current_interline_factor * splitting_factor;

  #ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "[SplittingSpace] Storage storage level " << lvl << ", DataSpace " << ds_idx
                  << ", Rank '" << unique_rank << "': Multi-rank splitting factor " << splitting_factor
                  << " from intraline to interline" << std::endl;
        std::cout << "  - intraline factor: " << current_intraline_factor << " -> " << new_intraline_factor
                  << " (divided by " << splitting_factor << ")" << std::endl;
        std::cout << "  - interline factor: " << current_interline_factor << " -> " << new_interline_factor
                  << " (multiplied by " << splitting_factor << ")" << std::endl;
  #endif

        // Apply the changes
        intraline_nest.factors[unique_rank] = new_intraline_factor;
        interline_nest.factors[unique_rank] = new_interline_factor;
      }
    }
  }


  // Apply PackingSpace choices (multi-rank packing: interline-to-intraline)
#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "[PackingSpace] Applying multi-rank packing for all dataspaces..." << std::endl;
#endif
  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++) {
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++) {
      uint64_t choice = packing_choice_per_lvl_per_ds[lvl][ds_idx];
      if (!(choice < multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx].size())){
#ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "Note: Do not need to pack for storage level " << lvl << ", dataspace " << ds_idx << " because no data could be fitted into a line capacity." << std::endl;
#endif
        continue;
      }
      const auto& multi_rank_option = multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx][choice];

      for (const auto& unique_rank : multi_rank_option.ranks)
      {
        uint32_t packing_factor = multi_rank_option.packing_factors.at(unique_rank);
        {
          std::stringstream ss;
          ss << "Layout vector size (" << layout_.size() << ") is smaller than or equal to current level index (" << lvl << ")";
          assert(layout_.size() > lvl && ss.str().c_str());
        }
        {
          std::stringstream ss;
          ss << "Intraline vector size (" << layout_[lvl].intraline.size() << ") is smaller than or equal to current dataspace index (" << ds_idx << ")";
          assert(layout_[lvl].intraline.size() > ds_idx && ss.str().c_str());
        }
        {
          std::stringstream ss;
          ss << "Interline vector size (" << layout_[lvl].interline.size() << ") is smaller than or equal to current dataspace index (" << ds_idx << ")";
          assert(layout_[lvl].interline.size() > ds_idx && ss.str().c_str());
        }
        // Get references to both nests for the specific dataspace
        auto& intraline_nest = layout_[lvl].intraline[ds_idx];
        auto& interline_nest = layout_[lvl].interline[ds_idx];
        
        if (intraline_nest.factors.find(unique_rank) == intraline_nest.factors.end() || interline_nest.factors.find(unique_rank) == interline_nest.factors.end())
        {
          Status error_status;
          error_status.success = false;
          error_status.fail_reason = "Rank " + unique_rank + " not found in intraline or interline nest for level " + std::to_string(lvl) + ", dataspace " + std::to_string(ds_idx);
          return {error_status};
        }

        // Get current factors from the layout (not from stored original values)
        uint32_t current_intraline_factor = (intraline_nest.factors.find(unique_rank) != intraline_nest.factors.end()
                                            ? intraline_nest.factors.at(unique_rank) : 1);
        uint32_t current_interline_factor = (interline_nest.factors.find(unique_rank) != interline_nest.factors.end()
                                            ? interline_nest.factors.at(unique_rank) : 1);

        // Validate that the packing factor divides the current interline factor
        if (current_interline_factor % packing_factor != 0)
        {
          Status error_status;
          error_status.success = false;
          error_status.fail_reason = "Multi-rank packing factor " + std::to_string(packing_factor) +
                                    " does not divide current interline factor " + std::to_string(current_interline_factor) +
                                    " for rank " + unique_rank + " at level " + std::to_string(lvl) + ", dataspace " + std::to_string(ds_idx);
          return {error_status};
        }

        // Pack the factor: move packing_factor from interline to intraline
        uint32_t new_intraline_factor = current_intraline_factor * packing_factor;
        uint32_t new_interline_factor = current_interline_factor / packing_factor;

  #ifdef DEBUG_CONSTRUCTION_LAYOUT
        std::cout << "[PackingSpace] Storage storage level " << lvl << ", DataSpace " << ds_idx
                  << ", Rank '" << unique_rank << "': Multi-rank packing factor " << packing_factor
                  << " from interline to intraline" << std::endl;
        std::cout << "  - intraline factor: " << current_intraline_factor << " -> " << new_intraline_factor
                  << " (multiplied by " << packing_factor << ")" << std::endl;
        std::cout << "  - interline factor: " << current_interline_factor << " -> " << new_interline_factor
                  << " (divided by " << packing_factor << ")" << std::endl;
  #endif

        // Apply the changes
        intraline_nest.factors[unique_rank] = new_intraline_factor;
        interline_nest.factors[unique_rank] = new_interline_factor;
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

    std::cout << "[AuthSpace] Storage storage level " << lvl << ", DataSpace " << ds_idx
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
        if (intraline_per_ds > storage_level_line_capacity[lvl]){
          throw std::runtime_error("Dataspace[" + std::to_string(ds_idx) + "] intraline size " + std::to_string(intraline_per_ds) + " exceeds storage level line capacity " + std::to_string(storage_level_line_capacity[lvl]) + " at level " + std::to_string(lvl));
        }
      }
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

  std::cout << "Step 2: Creating SplittingSpace and PackingSpace candidates from intraline factors..." << std::endl;

  // Clear previous design spaces
  splitting_options_per_level_.clear();
  splitting_choices_per_level_.clear();
  packing_options_per_level_.clear();
  packing_choices_per_level_.clear();
  multi_rank_splitting_options_per_level_per_ds_.clear();
  multi_rank_splitting_options_per_level_per_ds_.resize(num_storage_levels, std::vector<std::vector<MultiRankSplittingOption>>(num_data_spaces, std::vector<MultiRankSplittingOption>()));
  multi_rank_packing_options_per_level_per_ds_.clear();
  multi_rank_packing_options_per_level_per_ds_.resize(num_storage_levels, std::vector<std::vector<MultiRankPackingOption>>(num_data_spaces, std::vector<MultiRankPackingOption>()));

  // Phase 1: Get Memory Line size for all storage levels (What Layout Provide Per Cycle)
  std::vector<std::vector<std::uint64_t>> intraline_size_per_ds(num_storage_levels, std::vector<std::uint64_t>(num_data_spaces, 0));

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
      }
    }
  }

  // Phase 2: Check if the line capacity is sufficient for the intraline size
  // First, determine which levels require splitting (intraline_size > line_capacity)
  level_ds_requires_splitting_.resize(num_storage_levels, std::vector<bool>(num_data_spaces, false));
  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++){
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
      if(storage_level_line_capacity[lvl] < intraline_size_per_ds[lvl][ds_idx]){
        level_ds_requires_splitting_[lvl][ds_idx] = true;
        std::cout << "  storage level " << lvl << ": dataspace " << ds_idx << " requires splitting (intraline_size=" << intraline_size_per_ds[lvl][ds_idx]
                << " > line_capacity=" << storage_level_line_capacity[lvl] << ")" << std::endl;
      } else {
        level_ds_requires_splitting_[lvl][ds_idx] = false;
        std::cout << "  storage level " << lvl << ": dataspace " << ds_idx << " splitting optional (intraline_size=" << intraline_size_per_ds[lvl][ds_idx]
                  << " <= line_capacity=" << storage_level_line_capacity[lvl] << ")";
        if (storage_level_bypass_factor[lvl][ds_idx] == 0)
          std::cout << " - bypass" << std::endl;
        else
          std::cout << std::endl;
      }
    }
  }

  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++){
    // First analyze single-rank and multi-rank splitting possibilities for each dataspace.
    for(unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){ // single
      if(storage_level_line_capacity[lvl] < intraline_size_per_ds[lvl][ds_idx] && storage_level_bypass_factor[lvl][ds_idx]){
        // The product of all factors of intraline for a dataspace is too big to fit in the line capacity,
        // so need to reduce the factors of intraline by converting some factors into interline.

        std::cout << "  storage level " << lvl << ": dataspace " << ds_idx << " intraline_size (" << intraline_size_per_ds[lvl][ds_idx]
                  << ") exceeds line capacity (" << storage_level_line_capacity[lvl]
                  << "). Generating design space for factor conversions..." << std::endl;

        // Calculate maximum packing factor that can be applied
        uint32_t max_splitting_factor = static_cast<uint32_t>((static_cast<float>(intraline_size_per_ds[lvl][ds_idx]) + static_cast<float>(storage_level_line_capacity[lvl]) - 1) / static_cast<float>(storage_level_line_capacity[lvl]));
        if (max_splitting_factor > 1){
          std::cout << "    Maximum splitting factor: " << max_splitting_factor << std::endl;
          auto& intraline_nest = layout_.at(lvl).intraline.at(ds_idx);
          std::map<std::string, std::vector<uint32_t>> all_candidate_factors_per_rank;

          // First, analyze single-rank splitting possibilities for each rank.
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

              // Collect all possible splitting factors for each rank (for multi-rank combinations)
              if (!valid_factors.empty()) {
                all_candidate_factors_per_rank[rank] = valid_factors;
              }
            }
          }

          // Multi-rank splitting: find combinations of ranks that together can reduce intraline size to fit
          // Use all candidate factors (including those that don't fit individually)
          std::cout << "      Multi-rank splitting analysis:" << std::endl;
          std::vector<std::vector<std::string>> rank_combinations = GenerateRankCombinations(intraline_nest.ranks);

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
              multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx].push_back(multi_option);

              std::cout << "        --> Valid multi-rank option: Ranks [";
              for (size_t i = 0; i < rank_combo.size(); i++) {
                std::cout << rank_combo[i];
                if (i < rank_combo.size() - 1) std::cout << ", ";
              }
              std::cout << "], total_reduction for dataspace[" << ds_idx << "] = " << multi_option.total_reduction << std::endl;

              // Print individual splitting factors
              for (const auto& rank : rank_combo) {
                std::cout << "          " << rank << "(intraline): " << multi_option.original_intraline_factors.at(rank)
                          << " -> " << (multi_option.original_intraline_factors.at(rank) / multi_option.splitting_factors.at(rank))
                          << " (split by " << multi_option.splitting_factors.at(rank) << ")" << std::endl;
              }
            }
          }
        }
        else{
          std::cout << "  storage level " << lvl << ": splitting factor = 1, no splitting is needed." << std::endl;
        }

        if (multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx].empty()){
          std::cout << "  storage level " << lvl << ": dataspace " << ds_idx << " no multi-rank splitting options." << std::endl;
        }
        else{
          std::cout << "  storage level " << lvl << ": dataspace " << ds_idx << " has " <<  multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx].size() << " multi-rank splitting options." << std::endl;
        }
        // Calculate maximum splitting factor that can be applied
      }
      else if (storage_level_line_capacity[lvl] > intraline_size_per_ds[lvl][ds_idx] && storage_level_bypass_factor[lvl][ds_idx]){
        // Intraline has free space to hold more data, could convert some factors of interline into intraline,
        // this creates the overall design spaces

        std::cout << "  storage level " << lvl << " Dataspace[" << ds_idx << "] intraline_size (" << intraline_size_per_ds[lvl][ds_idx]
                  << ") has " << (storage_level_line_capacity[lvl] - intraline_size_per_ds[lvl][ds_idx])
                  << " free capacity. Generating design space for data packing..." << std::endl;

        // Calculate maximum packing factor that can be applied
        uint32_t max_packing_factor = static_cast<uint32_t>(static_cast<float>(storage_level_line_capacity[lvl]) / static_cast<float>(intraline_size_per_ds[lvl][ds_idx]));
        if (max_packing_factor > 1){
          std::cout << "    Maximum packing factor: " << max_packing_factor << std::endl;

          auto& inter_nest = layout_.at(lvl).interline.at(ds_idx);

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


            for (auto rank_combo = rank_combinations.rbegin(); rank_combo != rank_combinations.rend(); ++rank_combo) {
              MultiRankPackingOption multi_option;
              if (TestMultiRankPackingWithCandidates(lvl, ds_idx, *rank_combo, all_candidate_factors_per_rank,
                                                  intraline_size_per_ds, storage_level_line_capacity[lvl], multi_option)) {
                multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx].push_back(multi_option);

                std::cout << "      --> Valid multi-rank packing option: Ranks [";
                for (size_t i = 0; i < rank_combo->size(); i++) {
                  std::cout << (*rank_combo)[i];
                  if (i < rank_combo->size() - 1) std::cout << ", ";
                }
                std::cout << "], total_packing=" << multi_option.total_packing << std::endl;

                // Print individual packing factors
                for (const auto& rank : *rank_combo) {
                  std::cout << "        " << rank << "(interline): " << multi_option.original_interline_factors.at(rank)
                            << " -> " << (multi_option.original_interline_factors.at(rank) / multi_option.packing_factors.at(rank))
                            << " (pack by " << multi_option.packing_factors.at(rank) << ")" << std::endl;
                }
              }
            }
          } else {
            std::cout << "      Insufficient ranks for multi-rank combinations in this dataspace" << std::endl;
          }

        }
        else{
          std::cout << "  storage level " << lvl << ": packing factor = 1, no packing is needed." << std::endl;
        }

        if (multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx].empty()){
          std::cout << "  storage level " << lvl << ": dataspace " << ds_idx << " no multi-rank packing options." << std::endl;
        }
        else{
          std::cout << "  storage level " << lvl << ": dataspace " << ds_idx << " has " <<  multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx].size() << " multi-rank packing options." << std::endl;
        }
      }
      // Do nothing if the line capacity is equal to the intraline size
    }
  }

  // cross_dataspace_multi_rank_splitting_options_per_level_
  splitting_candidates_per_lvl_per_ds.clear();
  splitting_candidates_per_lvl_per_ds.resize(num_storage_levels, std::vector<std::uint64_t>(num_data_spaces, 1));
  splitting_candidates = 1;
  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++){
    std::cout << "  storage level " << lvl << " has " << multi_rank_splitting_options_per_level_per_ds_[lvl].size() << " multi-rank splitting options." << std::endl;
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
      if (multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx].empty()){
        splitting_candidates_per_lvl_per_ds[lvl][ds_idx] = 1;
      }
      else{
        splitting_candidates_per_lvl_per_ds[lvl][ds_idx] = multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx].size();
        splitting_candidates *= splitting_candidates_per_lvl_per_ds[lvl][ds_idx];
      }
    }
  }

  // Print flattened splitting choices in table format
  std::cout << "Breakdown of splitting candidates:" << std::endl;
  std::cout << "Level | DataSpace | Candidates" << std::endl;
  std::cout << "------|-----------|-----------" << std::endl;
  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++) {
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++) {
      std::cout << std::setw(6) << lvl << " | " 
                << std::setw(9) << ds_idx << " | "
                << std::setw(9) << splitting_candidates_per_lvl_per_ds[lvl][ds_idx] << std::endl;
    }
  }
  std::cout << std::endl;

  if (splitting_candidates == 1)
    splitting_candidates = 0;
  std::cout << "Splitting layout candidates: " << splitting_candidates << std::endl;

  // cross_dataspace_multi_rank_packing_options_per_level_
  packing_candidates_per_lvl_per_ds.clear();
  packing_candidates_per_lvl_per_ds.resize(num_storage_levels, std::vector<std::uint64_t>(num_data_spaces, 1));
  packing_candidates = 1;
  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++){
    std::cout << "  storage level " << lvl << " has " << multi_rank_packing_options_per_level_per_ds_[lvl].size() << " multi-rank packing options." << std::endl;
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
      if (multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx].empty()){
        packing_candidates_per_lvl_per_ds[lvl][ds_idx] = 1;
      }
      else{
        packing_candidates_per_lvl_per_ds[lvl][ds_idx] = multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx].size();
        packing_candidates *= packing_candidates_per_lvl_per_ds[lvl][ds_idx];
      }
    }
  }

  // Print flattened packing choices in table format
  std::cout << "Breakdown of packing candidates:" << std::endl;
  std::cout << "Level | DataSpace | Candidates" << std::endl;
  std::cout << "------|-----------|-----------" << std::endl;
  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++) {
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++) {
      std::cout << std::setw(6) << lvl << " | " 
                << std::setw(9) << ds_idx << " | "
                << std::setw(9) << packing_candidates_per_lvl_per_ds[lvl][ds_idx] << std::endl;
    }
  }
  std::cout << std::endl;

  if (packing_candidates == 1)
    packing_candidates = 0;
  std::cout << "Packing layout candidates: " << packing_candidates << std::endl;
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
      std::cout << "  storage level " << lvl << " has non-empty authblock_lines, will generate candidates" << std::endl;

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
              std::cout << "  Variable factor: storage level " << lvl
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
      std::cout << "  storage level " << lvl << " has empty authblock_lines, skipping candidate generation" << std::endl;
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

void Legal::SequentialFactorizeLayout(layout::Layouts& layout){
  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++)
  {
    std::cout << "lvl=" << lvl << " storage_level_line_capacity[lvl]=" << storage_level_line_capacity[lvl] << std::endl;
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++)
    {
      // Check if this dataspace is bypassed at this storage level
      uint32_t intraline_per_ds = 1;
      bool is_kept = storage_level_bypass_factor[lvl][ds_idx];

      if (is_kept)
      {
        auto intra_nest = layout.at(lvl).intraline.at(ds_idx);
        for (const auto &r : intra_nest.ranks) // Analyze slowdown per rank
        {
        int factor = (intra_nest.factors.find(r) != intra_nest.factors.end() ? intra_nest.factors.at(r) : 1);
          intraline_per_ds *= factor;
        }

        float splitting_factor = (float)intraline_per_ds / (float)storage_level_line_capacity[lvl];
        // Check if the intraline product of dataspaces is greater than the storage level line capacity
        std::cout << "Initial splitting_factor: " << splitting_factor << std::endl;
        for (const auto &r : layout.at(lvl).intraline.at(ds_idx).ranks)
        {
          std::cout << "  Processing rank " << r << ", current factor: " << layout.at(lvl).intraline.at(ds_idx).factors[r] << std::endl;
          if (layout.at(lvl).intraline.at(ds_idx).factors[r] > 1)
          {
            std::cout << "  rank: " << r << " intraline factor: " << layout.at(lvl).intraline.at(ds_idx).factors[r] << " -> 1 ";
            layout.at(lvl).interline.at(ds_idx).factors[r] *= layout.at(lvl).intraline.at(ds_idx).factors[r];
            splitting_factor = splitting_factor / (float)layout.at(lvl).intraline.at(ds_idx).factors[r];
            layout.at(lvl).intraline.at(ds_idx).factors[r] = 1;
            std::cout << " new splitting_factor: " << splitting_factor << std::endl;
          }
          if (splitting_factor < 1.0)
          {
            break;
          }
          std::cout << "Final splitting_factor for this iteration: " << splitting_factor << std::endl;
        }
      }
    }
  }
};

} // namespace layoutspace
