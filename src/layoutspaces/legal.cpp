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
#include <stdexcept>
#include <cassert>
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
    LayoutSpace(arch_specs, mapping, layout),
    split_id_(0),
    num_parent_splits_(0),
    legal_space_empty_(false)
{
  layout::PrintOverallLayout(layout);
  (void)skip_init; // Suppress unused parameter warning
  num_storage_levels = mapping.loop_nest.storage_tiling_boundaries.size();
  num_data_spaces = layout_.at(0).intraline.size();

  // Step 0: (1) Initialize the data bypass logic
  Init(arch_specs, mapping);

  // ToDo: Need to define a layout constraint.
  // Create a legal layoutspace when no layout configuration is provided
  /* Step 1: Create a concordant layout from the mapping, and assign values into layout_.
             temporal becomes interline; spatial becomes intraline.
             For imperfect factorization, the layout would just pad zeros to it.
  */

  /*  Step 2: Reform layout to be legal with a certain mapping.
               Define requested_parallelism (RP) as the product of extents of spatial loops.
               (1) all requested data should be fit into on-chip buffer. The legal space is empty if not.
               (2) all data being requested in parallel (spatial for loops) should fit in a single on-chip buffer line [line_cap(L)].
                 if RP == line_cap:
                   do nothing
                 if RP > line_cap:
                   further tiling #intraline and move some loops to interline (the tiling and which loops to move are the new design spaces)
                 if RP < line_cap:
                   pack all RP data first
                   tiling temporal loops to create more loop levels with smaller iteration counts.
                   enumerate temporal‑loop dimensions that can be packed into remaining slots
                   continue until buffer‑size constraint met
   */

  // print out mapping
  std::cout << "Mapping:" << std::endl;
  std::cout << "=====================" << std::endl;
  std::cout << mapping << std::endl;

  // Step 1: Create concordant layout from mapping
  CreateConcordantLayout(mapping);

  // Step 2: Check buffer capacity constraint
  // CheckBufferCapacityConstraint(arch_specs, mapping); // only need to be enabled if mapping does not prevent buffer overflow.
  CreateIntraLineSpace(arch_specs, mapping);

  // Step 3: CreateAuthSpace
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

  kept_data_spaces.clear();
  bypassed_data_spaces.clear();
  for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++)
  {
    auto storage_level_specs = arch_specs.topology.GetStorageLevel(storage_level);
#ifdef DEBUG_BUFFER_CAPACITY_CONSTRAINT
    std::cout << "    Level " << storage_level << " (" << storage_level_specs->name.Get() << "): ";
#endif

    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++)
    {
      // Check if this dataspace is bypassed at this storage level
      bool is_kept = mapping.datatype_bypass_nest.at(ds_idx).test(storage_level);

      if (is_kept)
      {
        if (ds_idx < layout_.at(storage_level).data_space.size())
          kept_data_spaces.push_back(layout_.at(storage_level).data_space[ds_idx]);
        else
          kept_data_spaces.push_back("DataSpace" + std::to_string(ds_idx));
      }
      else
      {
        if (ds_idx < layout_.at(storage_level).data_space.size())
          bypassed_data_spaces.push_back(layout_.at(storage_level).data_space[ds_idx]);
        else
          bypassed_data_spaces.push_back("DataSpace" + std::to_string(ds_idx));
      }
    }

#ifdef DEBUG_BUFFER_CAPACITY_CONSTRAINT
    std::cout << "Keep=[";
    for (size_t i = 0; i < kept_data_spaces.size(); i++)
    {
      std::cout << kept_data_spaces[i];
      if (i < kept_data_spaces.size() - 1) std::cout << ", ";
    }
    std::cout << "], Bypass=[";
    for (size_t i = 0; i < bypassed_data_spaces.size(); i++)
    {
      std::cout << bypassed_data_spaces[i];
      if (i < bypassed_data_spaces.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
#endif
  }

  // Step 0: (2) Initialize the storage level capacity vectors
  storage_level_total_capacity.resize(num_storage_levels, 0);
  storage_level_line_capacity.resize(num_storage_levels, 0);
   // Iterate through each storage level
   for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++)
   {
     auto storage_level_specs = arch_specs.topology.GetStorageLevel(storage_level);

     // Extract memory specifications
     std::uint64_t total_capacity = 0;
     std::uint64_t block_size = 0;
     std::uint64_t line_capacity = 0;
     double read_bandwidth = 0.0;
     double write_bandwidth = 0.0;

     // Extract block_size (elements per memory line)
     if (storage_level_specs->block_size.IsSpecified())
     {
       block_size = storage_level_specs->block_size.Get();
     }

     // Extract bandwidth specifications
     if (storage_level_specs->read_bandwidth.IsSpecified())
     {
       read_bandwidth = storage_level_specs->read_bandwidth.Get();
     }

     if (storage_level_specs->write_bandwidth.IsSpecified())
     {
       write_bandwidth = storage_level_specs->write_bandwidth.Get();
     }

     // Extract total capacity
     if (storage_level_specs->size.IsSpecified())
     {
       total_capacity = storage_level_specs->size.Get();
     }
     else
     {
       std::cout << "    WARNING: Storage level " << storage_level
                 << " (" << storage_level_specs->name.Get() << ") has unspecified size hence do it as infinite" << std::endl;
       total_capacity = std::numeric_limits<uint64_t>::max();
     }

     // Line capacity is the number of elements that can be accessed in parallel
     // Use block_size, and use the maximum of read_bandwidth and write_bandwidth, as fallback
     line_capacity = block_size;
     if (line_capacity == 0)
     {
       line_capacity = static_cast<std::uint64_t>(std::max(read_bandwidth, write_bandwidth));  // Fallback to bandwidth
     }

     // Store capacity values in member variables
     storage_level_total_capacity[storage_level] = static_cast<std::uint32_t>(total_capacity);
     storage_level_line_capacity[storage_level] = static_cast<std::uint32_t>(line_capacity);

 #ifdef DEBUG_BUFFER_CAPACITY_CONSTRAINT
     std::cout << "    Storage Level " << storage_level
               << " (" << storage_level_specs->name.Get() << "):" << std::endl;
     std::cout << "      Block size: " << block_size << " elements per line" << std::endl;
     std::cout << "      Total capacity: " << total_capacity << " elements" << std::endl;
     std::cout << "      Line capacity (bandwidth): " << line_capacity << " elements/cycle" << std::endl;
     std::cout << "      Read bandwidth: " << read_bandwidth << " elements/cycle" << std::endl;
     std::cout << "      Write bandwidth: " << write_bandwidth << " elements/cycle" << std::endl;
 #endif
    }
  }

  //
  // ConstructLayout() - Original version with ID parameter (delegates to three-parameter version)
  //
  std::vector<Status> Legal::ConstructLayout(ID layout_id_in, layout::Layouts* layouts, Mapping mapping, bool break_on_failure)
  {
    (void)break_on_failure; // Suppress unused parameter warning

    // This function delegates to the three-parameter version with default values
    // Convert ID to uint64_t
    uint128_t layout_int = layout_id_in.Integer();
    
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
  // ConstructLayout() - Three-parameter version with separate layout_id, layout_auth_id, and layout_packing_id
  //
  std::vector<Status> Legal::ConstructLayout(uint64_t layout_id, uint64_t layout_auth_id, uint64_t layout_packing_id, layout::Layouts* layouts, Mapping mapping, bool break_on_failure)
  {
  (void)break_on_failure; // Suppress unused parameter warning

  // This function takes separate IDs for all three design spaces:
  // - layout_id: for IntraLineSpace (intraline-to-interline conversions)
  // - layout_auth_id: for AuthSpace (authblock factor variations)
  // - layout_packing_id: for PackingSpace (interline-to-intraline packing)

  // Create a deep copy of the layout to ensure modifications don't affect the original
  CreateConcordantLayout(mapping);
  
#ifdef LAYOUT_CONSTRUCTION_DEBUG
  std::cout << "\n=== LAYOUT CONSTRUCTION START ===" << std::endl;
  std::cout << "Layout IDs: IntraLine=" << layout_id << ", Auth=" << layout_auth_id 
            << ", Packing=" << layout_packing_id << std::endl;
  std::cout << "Initial original layout:" << std::endl;
  layout::PrintOverallLayoutConcise(layout_);
#endif 

  layout::Layouts modified_layout = layout_;

  // If no variable factors, just return the original layout
  if (variable_authblock_factors_.empty() && variable_intraline_factors_.empty() && packing_options_per_level_.empty())
  {
    // Copy the current layout to the output parameter
    if (layouts != nullptr)
    {
      *layouts = modified_layout;
    }

    Status success_status;
    success_status.success = true;
    success_status.fail_reason = "";
    return {success_status};
  }

  // Validate individual design space IDs
  uint64_t authblock_candidates = 1;
  for (const auto& range : authblock_factor_ranges_)
  {
    authblock_candidates *= range.size();
  }

  uint64_t intraline_candidates = 1;
  for (const auto& range : intraline_conversion_ranges_)
  {
    intraline_candidates *= range.size();
  }

  uint64_t packing_candidates = 1;
  for (const auto& choices : packing_choices_per_level_)
  {
    packing_candidates *= choices;
  }

  // Validate layout_auth_id range
  if (!variable_authblock_factors_.empty() && layout_auth_id >= authblock_candidates)
  {
    Status error_status;
    error_status.success = false;
    error_status.fail_reason = "layout_auth_id " + std::to_string(layout_auth_id) + " exceeds AuthSpace size " + std::to_string(authblock_candidates);
    return {error_status};
  }

  // Validate layout_id range
  if (!variable_intraline_factors_.empty() && layout_id >= intraline_candidates)
  {
    Status error_status;
    error_status.success = false;
    error_status.fail_reason = "layout_id " + std::to_string(layout_id) + " exceeds IntraLineSpace size " + std::to_string(intraline_candidates);
    return {error_status};
  }

  // Calculate PackingSpace design space size
  for (const auto& choices : packing_choices_per_level_)
  {
    packing_candidates *= choices;
  }

  // Validate layout_packing_id range
  if (!packing_options_per_level_.empty() && layout_packing_id >= packing_candidates)
  {
    Status error_status;
    error_status.success = false;
    error_status.fail_reason = "layout_packing_id " + std::to_string(layout_packing_id) + " exceeds PackingSpace size " + std::to_string(packing_candidates);
    return {error_status};
  }

  // Decode AuthSpace factor choices using layout_auth_id
  std::vector<uint32_t> authblock_choices(variable_authblock_factors_.size());
  std::uint64_t remaining_auth_id = layout_auth_id;

  for (size_t i = 0; i < variable_authblock_factors_.size(); i++)
  {
    const auto& divisors = authblock_factor_ranges_[i];
    uint32_t divisor_index = remaining_auth_id % divisors.size();
    authblock_choices[i] = divisors[divisor_index];
    remaining_auth_id /= divisors.size();
  }

  // Decode IntraLineSpace conversion choices using layout_id
  std::vector<uint32_t> intraline_choices(variable_intraline_factors_.size());
  std::uint64_t remaining_intraline_id = layout_id;

  for (size_t i = 0; i < variable_intraline_factors_.size(); i++)
  {
    const auto& conversions = intraline_conversion_ranges_[i];
    uint32_t conversion_index = remaining_intraline_id % conversions.size();
    intraline_choices[i] = conversions[conversion_index];
    remaining_intraline_id /= conversions.size();
  }

  // Decode PackingSpace choices using layout_packing_id (per-level approach)
  std::vector<uint32_t> packing_choice_per_level(packing_options_per_level_.size());
  std::uint64_t remaining_packing_id = layout_packing_id;

  for (size_t level = 0; level < packing_options_per_level_.size(); level++)
  {
    uint32_t choice_index = remaining_packing_id % packing_choices_per_level_[level];
    packing_choice_per_level[level] = choice_index;
    remaining_packing_id /= packing_choices_per_level_[level];
  }

#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "Constructing layout with three separate IDs:" << std::endl;
  std::cout << "  AuthSpace (layout_auth_id): " << layout_auth_id << ", choices: [";
  for (size_t i = 0; i < authblock_choices.size(); i++)
  {
    std::cout << authblock_choices[i];
    if (i < authblock_choices.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << "  IntraLineSpace (layout_id): " << layout_id << ", choices: [";
  for (size_t i = 0; i < intraline_choices.size(); i++)
  {
    std::cout << intraline_choices[i];
    if (i < intraline_choices.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << "  PackingSpace (layout_packing_id): " << layout_packing_id << ", per-level choices: [";
  for (size_t level = 0; level < packing_choice_per_level.size(); level++)
  {
    std::cout << "L" << level << ":" << packing_choice_per_level[level];
    if (level < packing_choice_per_level.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
#endif

  // Apply AuthSpace factor choices (using layout_auth_id)
  for (size_t i = 0; i < variable_authblock_factors_.size(); i++)
  {
    auto& var_factor = variable_authblock_factors_[i];
    unsigned lvl = std::get<0>(var_factor);
    unsigned ds_idx = std::get<1>(var_factor);
    std::string rank = std::get<2>(var_factor);
    uint32_t chosen_factor = authblock_choices[i];

    // Apply the chosen factor to the authblock_lines nest
    auto& authblock_nest = modified_layout[lvl].authblock_lines[ds_idx];

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
    
#ifdef LAYOUT_CONSTRUCTION_DEBUG
    // Get the old factor for comparison
    uint32_t old_authblock_factor = (authblock_nest.factors.find(rank) != authblock_nest.factors.end()
                                    ? authblock_nest.factors.at(rank) : 1);
    
    std::cout << "[AuthSpace] Storage Level " << lvl << ", DataSpace " << ds_idx 
              << ", Rank '" << rank << "': authblock_lines factor " 
              << old_authblock_factor << " -> " << chosen_factor << std::endl;
#endif
  }

  // Apply IntraLineSpace conversion choices (using layout_id)
  for (size_t i = 0; i < variable_intraline_factors_.size(); i++)
  {
    auto& var_factor = variable_intraline_factors_[i];
    unsigned lvl = std::get<0>(var_factor);
    unsigned ds_idx = std::get<1>(var_factor);
    std::string rank = std::get<2>(var_factor);
    uint32_t original_factor = std::get<3>(var_factor);
    uint32_t conversion_factor = intraline_choices[i];

    // Validate indices
    if (lvl >= modified_layout.size())
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Invalid storage level " + std::to_string(lvl) + " in intraline variable factor";
      return {error_status};
    }

    if (ds_idx >= modified_layout[lvl].intraline.size())
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Invalid data space index " + std::to_string(ds_idx) + " in intraline variable factor";
      return {error_status};
    }

    // Apply the intraline-to-interline conversion
    auto& intraline_nest = modified_layout[lvl].intraline[ds_idx];
    auto& interline_nest = modified_layout[lvl].interline[ds_idx];

    // Check if rank exists in both nests
    auto intra_rank_it = std::find(intraline_nest.ranks.begin(), intraline_nest.ranks.end(), rank);
    auto inter_rank_it = std::find(interline_nest.ranks.begin(), interline_nest.ranks.end(), rank);

    if (intra_rank_it == intraline_nest.ranks.end() || inter_rank_it == interline_nest.ranks.end())
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Rank " + rank + " not found in intraline or interline nest for level " + std::to_string(lvl) + ", dataspace " + std::to_string(ds_idx);
      return {error_status};
    }

    // Move the conversion factor from intraline to interline
    uint32_t new_intraline_factor = original_factor / conversion_factor;
    uint32_t current_interline_factor = (interline_nest.factors.find(rank) != interline_nest.factors.end()
                                        ? interline_nest.factors.at(rank) : 1);
    uint32_t new_interline_factor = current_interline_factor * conversion_factor;

#ifdef LAYOUT_CONSTRUCTION_DEBUG
    std::cout << "[IntraLineSpace] Storage Level " << lvl << ", DataSpace " << ds_idx 
              << ", Rank '" << rank << "': Converting factor " << conversion_factor 
              << " from intraline to interline" << std::endl;
    std::cout << "  - intraline factor: " << original_factor << " -> " << new_intraline_factor 
              << " (divided by " << conversion_factor << ")" << std::endl;
    std::cout << "  - interline factor: " << current_interline_factor << " -> " << new_interline_factor 
              << " (multiplied by " << conversion_factor << ")" << std::endl;
#endif

    intraline_nest.factors[rank] = new_intraline_factor;
    interline_nest.factors[rank] = new_interline_factor;
  }

  // Apply PackingSpace choices (one rank per storage level)
  std::cout << "[PackingSpace] Applying single-rank-per-level packing..." << std::endl;
  
  for (size_t level = 0; level < packing_choice_per_level.size(); level++)
  {
    uint32_t choice_index = packing_choice_per_level[level];
    
    // Choice 0 means "no packing" for this level
    if (choice_index == 0)
    {
      std::cout << "[PackingSpace] Storage Level " << level << ": No packing applied" << std::endl;
      continue;
    }
    
    // Choice > 0 means apply the corresponding packing option
    if (choice_index > packing_options_per_level_[level].size())
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Invalid packing choice " + std::to_string(choice_index) + " for level " + std::to_string(level);
      return {error_status};
    }
    
    // Get the selected packing option (subtract 1 since choice 0 is "no packing")
    const auto& packing_option = packing_options_per_level_[level][choice_index - 1];
    
    unsigned ds_idx = packing_option.dataspace;
    std::string rank = packing_option.rank;
    uint32_t original_interline_factor = packing_option.original_interline_factor;
    uint32_t packing_factor = packing_option.packing_factor;

    // Validate indices
    if (level >= modified_layout.size())
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Invalid storage level " + std::to_string(level) + " in packing option";
      return {error_status};
    }

    if (ds_idx >= modified_layout[level].intraline.size())
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Invalid data space index " + std::to_string(ds_idx) + " in packing option";
      return {error_status};
    }

    // Apply the interline-to-intraline packing
    auto& intraline_nest = modified_layout[level].intraline[ds_idx];
    auto& interline_nest = modified_layout[level].interline[ds_idx];

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

  #ifdef LAYOUT_CONSTRUCTION_DEBUG
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

  // Copy the modified layout to the output parameter
  if (layouts != nullptr)
  {
    *layouts = modified_layout;
  }

  std::cout << "\n=== LAYOUT CONSTRUCTION COMPLETE ===" << std::endl;
  std::cout << "Final modified layout:" << std::endl;
  layout::PrintOverallLayoutConcise(modified_layout);

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
  std::cout << "Step 1: Create Concordant Layout..." << std::endl;
  std::cout << "Total number of storage levels: " << mapping.loop_nest.storage_tiling_boundaries.size() << std::endl;
  std::cout << "Total number of layout levels: " << layout_.size() << std::endl;
  assert(mapping.loop_nest.storage_tiling_boundaries.size() == layout_.size());
  std::cout << "Total number of data spaces: " << layout_.at(0).intraline.size() << std::endl;

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

  std::vector<std::vector<uint32_t>> tensor_size;
  tensor_size.resize(num_storage_levels, std::vector<uint32_t>(num_data_spaces, 0));

  // first level: number storage levels
  // second level: number data spaces
  // third level: size of tensor
  for (unsigned lvl=0; lvl < num_storage_levels; lvl++){
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
      uint32_t dataspace_size_cur_lvl = 1;
      for (auto & rank: layout_.at(lvl).intraline.at(ds_idx).ranks){
        // Get dimension IDs for this rank
        const auto& dim_ids = layout_.at(lvl).rankToFactorizedDimensionID.at(rank);

        // Calculate rank size using cumulative dimension values with coefficients
        uint32_t rank_size = 0;
        const auto& coefficient = layout_.at(lvl).rankToCoefficientValue[rank];
        for (unsigned idx=0; idx < dim_ids.size(); idx++){
          auto cumulative_it = cumulatively_product_dimval[lvl].find(dim_ids[idx]);
          if (cumulative_it != cumulatively_product_dimval[lvl].end())
          {
            auto dim_value = cumulative_it->second;
            if (dim_ids.size() > 1){
              if (dim_value == 1){
                if (idx < dim_ids.size()-1){
                  rank_size += dim_value;
                }
              }
              else{
                if (idx < dim_ids.size()-1){
                  rank_size += dim_value*coefficient[idx];
                }
                else{
                  rank_size += dim_value*coefficient[idx] - 1;
                }
              }
            }else{
              rank_size += dim_value;
            }
          }
        }
        dataspace_size_cur_lvl *= rank_size;
      }
      tensor_size[lvl][ds_idx] = dataspace_size_cur_lvl;
    }
  }

  // Print out the tensor size
#ifdef LAYOUT_CONSTRUCTION_DEBUG
  for (unsigned lvl=0; lvl < tensor_size.size(); lvl++){
    std::cout << "For a specific storage level " << lvl << ", the tensor size is: ";
    for (unsigned ds_idx = 0; ds_idx < tensor_size[lvl].size(); ds_idx++){
      std::cout << tensor_size[lvl][ds_idx] << " ";
    }
    std::cout << std::endl;
  }
#endif
}

//
// CreateIntraLineSpace() - Step 3: Generate all possible intraline factor combinations
//
void Legal::CreateIntraLineSpace(model::Engine::Specs arch_specs, const Mapping& mapping)
{
  (void) arch_specs; // Suppress unused parameter warning

  std::cout << "Step 2: Creating layout candidate space from intraline factors..." << std::endl;

  // Clear previous design spaces
  variable_intraline_factors_.clear();
  intraline_conversion_ranges_.clear();
  packing_options_per_level_.clear();
  packing_choices_per_level_.clear();

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
  for (unsigned lvl = 0; lvl < num_storage_levels; lvl++){
    if(storage_level_line_capacity[lvl] < intraline_size_per_lvl[lvl]){
      // The product of all factors of intraline for a dataspace is too big to fit in the line capacity,
      // so need to reduce the factors of intraline by converting some factors into interline.

      std::cout << "  Level " << lvl << ": intraline_size (" << intraline_size_per_lvl[lvl]
                << ") exceeds line capacity (" << storage_level_line_capacity[lvl]
                << "). Generating design space for factor conversions..." << std::endl;

      // For each dataspace, analyze conversion possibilities and store in member variables
      for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
        auto& intra_nest = layout_.at(lvl).intraline.at(ds_idx);
        std::cout << "    DataSpace " << ds_idx << ":" << std::endl;

        // Single-rank conversions: find minimal factors that can solve the overflow
        for (const auto& rank : intra_nest.ranks) {
          uint32_t current_factor = (intra_nest.factors.find(rank) != intra_nest.factors.end()
                                    ? intra_nest.factors.at(rank) : 1);

          if (current_factor > 1) {
            std::vector<uint32_t> divisors = FindDivisors(current_factor);
            std::vector<uint32_t> valid_conversions;

            // Test each divisor (excluding 1) to see if it solves the overflow
            for (uint32_t divisor : divisors) {
              if (divisor > 1) { // Skip 1 as it means no conversion
                // Calculate new intraline_size if this divisor is moved to interline
                std::vector<uint64_t> new_intraline_size_per_ds(num_data_spaces, 0);
                for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
                  new_intraline_size_per_ds[ds_idx] = intraline_size_per_ds[lvl][ds_idx] / divisor;
                }
                uint64_t new_intraline_size = std::accumulate(new_intraline_size_per_ds.begin(), new_intraline_size_per_ds.end(), 0);

                if (new_intraline_size <= storage_level_line_capacity[lvl]) {
                  // This divisor solves the overflow - add to design space
                  valid_conversions.push_back(divisor);
                  std::cout << "      Rank " << rank << ": moving factor " << divisor
                            << " to interline gives intraline_size=" << new_intraline_size
                            << " (fits in capacity)" << std::endl;

                  // For minimal conversion, we prefer the smallest factor that works
                  break;
                } else {
                  std::cout << "      Rank " << rank << ": moving factor " << divisor
                            << " to interline gives intraline_size=" << new_intraline_size
                            << " (still exceeds capacity)" << std::endl;
                }
              }
            }

            // If no single divisor works, add all divisors for potential multi-rank combinations
            if (valid_conversions.empty()) {
              std::cout << "      Rank " << rank << ": no single factor conversion works, "
                        << "adding all divisors for multi-rank combinations" << std::endl;
              for (uint32_t divisor : divisors) {
                if (divisor > 1) {
                  valid_conversions.push_back(divisor);
                }
              }
            }

            // Store in member variables for later use by ConstructLayout
            if (!valid_conversions.empty()) {
              variable_intraline_factors_.push_back(std::make_tuple(lvl, ds_idx, rank, current_factor));
              intraline_conversion_ranges_.push_back(valid_conversions);

              std::cout << "      Stored variable factor: Level " << lvl
                       << ", DataSpace " << ds_idx << ", Rank " << rank
                       << ", original_factor: " << current_factor
                       << ", conversion_options: [";
              for (size_t i = 0; i < valid_conversions.size(); i++) {
                std::cout << valid_conversions[i];
                if (i < valid_conversions.size() - 1) std::cout << ", ";
              }
              std::cout << "]" << std::endl;
            }
          }
        }
      }
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
                  for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
                    new_intraline_size_per_ds[ds_idx] = intraline_size_per_ds[lvl][ds_idx] * divisor;
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

  // Print summary of intraline design space
  if (!variable_intraline_factors_.empty()) {
    std::cout << "  Total intraline conversion variables: " << variable_intraline_factors_.size() << std::endl;

    // Calculate number of intraline conversion candidates
    uint64_t intraline_candidates = 1;
    for (const auto& range : intraline_conversion_ranges_) {
      intraline_candidates *= range.size();
    }
    std::cout << "  Intraline conversion layout candidates: " << intraline_candidates << std::endl;
  }

  // Setup choices per level and print summary of packing design space
  if (!packing_options_per_level_.empty()) {
    // Set up the number of choices for each level (including "no packing" option)
    packing_choices_per_level_.resize(packing_options_per_level_.size());
    for (size_t level = 0; level < packing_options_per_level_.size(); level++) {
      // +1 for "no packing" option (choice 0)
      packing_choices_per_level_[level] = packing_options_per_level_[level].size() + 1;
    }

    // Calculate total number of packing candidates
    uint64_t packing_candidates = 1;
    for (const auto& choices : packing_choices_per_level_) {
      packing_candidates *= choices;
    }

    std::cout << "  Total packing options across all levels: ";
    uint64_t total_options = 0;
    for (const auto& level_options : packing_options_per_level_) {
      total_options += level_options.size();
    }
    std::cout << total_options << std::endl;
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
        // const auto& interline_nest = layout_.at(lvl).interline.at(ds_idx);
        // const auto& intraline_nest = layout_.at(lvl).intraline.at(ds_idx);

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
            for (uint32_t dim_id : dims)
            {
              auto cumulative_it_lvl = cumulatively_product_dimval[lvl-1].find(dim_id);
              auto cumulative_it_lvl_minus_1 = cumulatively_product_dimval[lvl-2].find(dim_id);

              if (cumulative_it_lvl != cumulatively_product_dimval[lvl-1].end() &&
                  cumulative_it_lvl_minus_1 != cumulatively_product_dimval[lvl-2].end())
              {
                uint32_t ratio = cumulative_it_lvl->second / cumulative_it_lvl_minus_1->second;
                max_factor *= ratio;
              }
              else
              {
                std::cout << "Warning: dimension ID " << dim_id << " not found in cumulatively_product_dimval for level " << lvl-1 << " or " << (lvl-2) << std::endl;
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
  uint64_t authblock_candidates = 1;
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

  // Calculate total number of combinations from intraline conversions
  uint64_t intraline_candidates = 1;
  if (!variable_intraline_factors_.empty())
  {
    for (const auto& range : intraline_conversion_ranges_)
    {
      intraline_candidates *= range.size();
    }
    std::cout << "  Intraline conversion layout candidates: " << intraline_candidates << std::endl;
  }
  else
  {
    std::cout << "  No variable intraline conversion factors found." << std::endl;
  }

  // Calculate total number of combinations from packing factors
  uint64_t packing_candidates = 1;
  if (!packing_options_per_level_.empty())
  {
    for (const auto& choices : packing_choices_per_level_)
    {
      packing_candidates *= choices;
    }
    std::cout << "  Packing layout candidates: " << packing_candidates << std::endl;
  }
  else
  {
    std::cout << "  No variable packing factors found." << std::endl;
  }

  // Total layout candidates is the product of all three design spaces
  num_layout_candidates = authblock_candidates * intraline_candidates * packing_candidates;
  std::cout << "  Total combined layout candidates: " << num_layout_candidates << std::endl;
  std::cout << "  Variable factors count: " << variable_authblock_factors_.size() << std::endl;
  std::cout << "  Note: Only using divisors of max_factor for each variable factor" << std::endl;
  std::cout << "  ✓ Layout candidate space created successfully" << std::endl;
}

} // namespace layoutspace
