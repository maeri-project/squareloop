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

  // Step 0: (1) Initialize the data bypass logic
  if (!skip_init)
  {
    Init(arch_specs, mapping);
  }

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

  // Step 3: CreateSpace
  CreateSpace(arch_specs);

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
  unsigned num_storage_levels = mapping.loop_nest.storage_tiling_boundaries.size();
  unsigned num_data_spaces = layout_.at(0).intraline.size();
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
// ConstructLayout()
//
std::vector<Status> Legal::ConstructLayout(ID layout_id, layout::Layouts* layouts, bool break_on_failure)
{
  (void)break_on_failure; // Suppress unused parameter warning

  // If no variable factors, just return the original layout
  if (variable_authblock_factors_.empty())
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

  // Decode the layout ID into component choices
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

  // Validate layout ID range
  if (linear_id >= num_layout_candidates)
  {
    Status error_status;
    error_status.success = false;
    error_status.fail_reason = "Layout ID " + std::to_string(linear_id) + " exceeds candidate space size " + std::to_string(num_layout_candidates);
    return {error_status};
  }

  // Decode the linear ID into individual factor choices using modular arithmetic
  std::vector<uint32_t> factor_choices(variable_authblock_factors_.size());
  std::uint64_t remaining_id = linear_id;

  for (size_t i = 0; i < variable_authblock_factors_.size(); i++)
  {
    const auto& divisors = authblock_factor_ranges_[i];
    uint32_t divisor_index = remaining_id % divisors.size();
    factor_choices[i] = divisors[divisor_index]; // Use actual divisor value
    remaining_id /= divisors.size();
  }

#ifdef DEBUG_CONSTRUCTION_LAYOUT
  std::cout << "Constructing layout ID " << linear_id << " with factor choices: [";
  for (size_t i = 0; i < factor_choices.size(); i++)
  {
    std::cout << factor_choices[i];
    if (i < factor_choices.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
#endif

  // Create a copy of the current layout to modify
  layout::Layouts modified_layout = layout_;

  // Apply the decoded factor choices to create the specific layout
  for (size_t i = 0; i < variable_authblock_factors_.size(); i++)
  {
    auto& var_factor = variable_authblock_factors_[i];
    unsigned lvl = std::get<0>(var_factor);
    unsigned ds_idx = std::get<1>(var_factor);
    std::string rank = std::get<2>(var_factor);
    uint32_t chosen_factor = factor_choices[i];

    // Validate indices
    if (lvl >= modified_layout.size())
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Invalid storage level " + std::to_string(lvl) + " in variable factor";
      return {error_status};
    }

    if (ds_idx >= modified_layout[lvl].authblock_lines.size())
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = "Invalid data space index " + std::to_string(ds_idx) + " in variable factor";
      return {error_status};
    }

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

#ifdef DEBUG_CONSTRUCTION_LAYOUT
    std::cout << "  Applied factor " << chosen_factor << " to level " << lvl
              << ", dataspace " << ds_idx << ", rank " << rank << std::endl;
#endif
  }

  // Copy the modified layout to the output parameter
  if (layouts != nullptr)
  {
    *layouts = modified_layout;
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
  unsigned num_loops = mapping.loop_nest.loops.size();
  unsigned num_storage_levels = mapping.loop_nest.storage_tiling_boundaries.size();
  unsigned num_data_spaces = layout_.at(0).intraline.size();
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
        uint32_t total = 0;
        if (dim_ids.size() > 1){
          const auto& coefficient = layout_.at(lvl).rankToCoefficientValue[rank];
          for (unsigned idx=0; idx < dim_ids.size(); idx++){
            auto dim_value = cumulatively_intraline_dimval[lvl][dim_ids[idx]];
            if (idx == dim_ids.size()-1){
              if (dim_value == 1){
                total +=  dim_value - 1;
              }else{
                total +=  dim_value*coefficient[idx] - 1;
              }
            }
            else{
              if (dim_value == 1){
                total +=  dim_value;
              }else{
                total +=  dim_value*coefficient[idx];
              }
            }
#ifdef DEBUG_CONCORDANT_LAYOUT
            std::cout << "dim_value=" << dim_value << "--coef[" << idx << "]=" << coefficient[idx] << "; ";
#endif
          }
        }
        else{
          auto dim_value = cumulatively_intraline_dimval[lvl][dim_ids[0]];
          total = dim_value;
        }

        layout_.at(lvl).intraline.at(i).factors.at(rank) = total;
#ifdef DEBUG_CONCORDANT_LAYOUT
        std::cout << "level=" << lvl << " i=" << i << " rank=" << rank << " intraline = " << total << std::endl;
#endif

        total = 0;
        if (dim_ids.size() > 1){
          const auto& coefficient = layout_.at(lvl).rankToCoefficientValue[rank];
          for (unsigned idx=0; idx < dim_ids.size(); idx++){
            auto dim_value = (cumulatively_product_dimval[lvl][dim_ids[idx]] + cumulatively_intraline_dimval[lvl][dim_ids[idx]] - 1) / cumulatively_intraline_dimval[lvl][dim_ids[idx]];
            if (idx == dim_ids.size()-1){
              if (dim_value == 1){
                total +=  dim_value - 1;
              }else{
                total +=  dim_value*coefficient[idx] - 1;
              }
            }
            else{
              if (dim_value == 1){
                total +=  dim_value;
              }else{
                total +=  dim_value*coefficient[idx];
              }
            }
          }
        }
        else{
          auto dim_value = (cumulatively_product_dimval[lvl][dim_ids[0]] + cumulatively_intraline_dimval[lvl][dim_ids[0]] - 1) / cumulatively_intraline_dimval[lvl][dim_ids[0]];
          total = dim_value;
        }

        layout_.at(lvl).interline.at(i).factors.at(rank) = total;
#ifdef DEBUG_CONCORDANT_LAYOUT
        std::cout << "level=" << lvl << " i=" << i << " rank=" << rank << " interline = " << total << std::endl;
#endif
      }
    }
  }

#ifdef DEBUG_CONCORDANT_LAYOUT
  layout::PrintOverallLayout(layout_);

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
        if (dim_ids.size() > 1){
          const auto& coefficient = layout_.at(lvl).rankToCoefficientValue[rank];
          for (unsigned idx=0; idx < dim_ids.size(); idx++){
            auto cumulative_it = cumulatively_product_dimval[lvl].find(dim_ids[idx]);
            if (cumulative_it != cumulatively_product_dimval[lvl].end())
            {
              auto dim_value = cumulative_it->second;
              if (idx == dim_ids.size()-1){
                if (dim_value == 1){
                  rank_size += dim_value - 1;
                }else{
                  rank_size += dim_value*coefficient[idx] - 1;
                }
              }
              else{
                if (dim_value == 1){
                  rank_size += dim_value;
                }else{
                  rank_size += dim_value*coefficient[idx];
                }
              }
            }
          }
        }
        else{
          auto cumulative_it = cumulatively_product_dimval[lvl].find(dim_ids[0]);
          if (cumulative_it != cumulatively_product_dimval[lvl].end())
          {
            rank_size = cumulative_it->second;
          }
          else
          {
            rank_size = 1;
          }
        }
        
        dataspace_size_cur_lvl *= rank_size;
      }
      tensor_size[lvl][ds_idx] = dataspace_size_cur_lvl;
    }
  }

  // Print out the tensor size
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
// CheckBufferCapacityConstraint() - Check if data fits in buffer
//
bool Legal::CheckBufferCapacityConstraint(model::Engine::Specs arch_specs, const Mapping& mapping)
{

#ifdef DEBUG_BUFFER_CAPACITY_CONSTRAINT
  std::cout << "  Checking buffer capacity constraints..." << std::endl;
#endif
  unsigned num_storage_levels = arch_specs.topology.NumStorageLevels();
  unsigned num_data_spaces = layout_.at(0).intraline.size();

  for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++)
  {
  // Get cumulative dimension values for this storage level
    std::uint64_t total_capacity = storage_level_total_capacity[storage_level];
    std::uint64_t line_capacity = storage_level_line_capacity[storage_level];
    const auto& level_dimval = cumulatively_product_dimval[storage_level];

    // Calculate total data requirements at this storage level
    std::uint64_t total_data_size = 0;
    std::uint64_t total_parallel_accesses = 0;

    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++)
    {
      // Check if this dataspace is bypassed at this storage level
      bool is_kept = mapping.datatype_bypass_nest.at(ds_idx).test(storage_level);

      std::uint64_t dataspace_total_size = 1;    // Total data size for this dataspace
      std::uint64_t dataspace_parallel_size = 1; // Parallel access requirement for this dataspace

      if (is_kept)
      {
        // Calculate data size using layout factors and cumulative dimension values
        const auto& ranks = layout_.at(storage_level).intraline.at(ds_idx).ranks;

        for (auto& rank : ranks)
        {
          auto intraline_factor = layout_.at(storage_level).intraline.at(ds_idx).factors[rank];
          auto interline_factor = layout_.at(storage_level).interline.at(ds_idx).factors[rank];

          // Get dimension IDs for this rank
          const auto& dim_ids = layout_.at(storage_level).rankToFactorizedDimensionID.at(rank);

          // Calculate rank size using cumulative dimension values with coefficients
          std::uint64_t rank_dimension_product = 0;
          if (dim_ids.size() > 1){
            const auto& coefficient = layout_.at(storage_level).rankToCoefficientValue[rank];
            for (unsigned idx=0; idx < dim_ids.size(); idx++){
              if (level_dimval.find(dim_ids[idx]) != level_dimval.end())
              {
                auto dim_value = level_dimval.at(dim_ids[idx]);
                if (idx == dim_ids.size()-1){
                  if (dim_value == 1){
                    rank_dimension_product += dim_value - 1;
                  }else{
                    rank_dimension_product += dim_value*coefficient[idx] - 1;
                  }
                }
                else{
                  if (dim_value == 1){
                    rank_dimension_product += dim_value;
                  }else{
                    rank_dimension_product += dim_value*coefficient[idx];
                  }
                }
              }
            }
          }
          else{
            if (level_dimval.find(dim_ids[0]) != level_dimval.end())
            {
              rank_dimension_product = level_dimval.at(dim_ids[0]);
            }
            else
            {
              rank_dimension_product = 1;
            }
          }

          // The actual rank size is constrained by the layout factors and dimension values
          std::uint64_t rank_total_size = std::min(rank_dimension_product,
                                                  static_cast<std::uint64_t>(intraline_factor * interline_factor));
          std::uint64_t rank_parallel_size = std::min(rank_dimension_product,
                                                     static_cast<std::uint64_t>(intraline_factor));

          // Accumulate across ranks (multiplicative since ranks are dimensions)
          dataspace_total_size *= rank_total_size;
          dataspace_parallel_size *= rank_parallel_size;
        }

        // Handle case with no ranks (scalar data)
        if (ranks.empty())
        {
          dataspace_total_size = 1;
          dataspace_parallel_size = 1;
        }

        // Add to total requirements across all data spaces (only if not bypassed)
        total_data_size += dataspace_total_size;
        total_parallel_accesses += dataspace_parallel_size;
      }
      else
      {
        // Bypassed data spaces don't contribute to capacity requirements
        dataspace_total_size = 0;
        dataspace_parallel_size = 0;
      }
#ifdef DEBUG_BUFFER_CAPACITY_CONSTRAINT
      std::cout << "      Data space " << ds_idx;
      if (ds_idx < layout_.at(storage_level).data_space.size())
      {
        std::cout << " (" << layout_.at(storage_level).data_space[ds_idx] << ")";
      }
      std::cout << (is_kept ? " [KEPT]" : " [BYPASSED]") << ":" << std::endl;
      std::cout << "        Total data size: " << dataspace_total_size << " elements" << std::endl;
      std::cout << "        Parallel accesses: " << dataspace_parallel_size << " elements" << std::endl;
#endif
    }

#ifdef DEBUG_BUFFER_CAPACITY_CONSTRAINT
    std::cout << "      TOTAL data size across all spaces: " << total_data_size << " elements" << std::endl;
    std::cout << "      TOTAL parallel accesses: " << total_parallel_accesses << " elements" << std::endl;
#endif

    // Check capacity constraint: total data size should not exceed buffer capacity
    if (total_data_size > total_capacity)
      throw std::runtime_error("Buffer capacity constraint violation: Total data size (" + std::to_string(total_data_size) + ") exceeds memory capacity (" + std::to_string(total_capacity) + ")");

    // Check bandwidth constraint: parallel accesses must fit in line capacity
    if (total_parallel_accesses > line_capacity)
      throw std::runtime_error(" Buffer bandwidth constraint violation: Total parallel accesses (" + std::to_string(total_parallel_accesses) + ") exceed line capacity (" + std::to_string(line_capacity) + ")");

    std::cout << "    ✓ Storage level " << storage_level << " has sufficient (1) space and (2) bandwidth" << std::endl;
  }

  std::cout << "  ✓ All buffer capacity constraints satisfied" << std::endl;

  // Print storage level capacities
  std::cout << "Storage level capacities:" << std::endl;
  for (uint32_t i = 0; i < storage_level_total_capacity.size(); i++) {
    std::cout << "  Level " << i << ":" << std::endl;
    std::cout << "    Total capacity: " << storage_level_total_capacity[i] << std::endl;
    std::cout << "    Line capacity: " << storage_level_line_capacity[i] << std::endl;
  }
  return true;
}

//
// CreateSpace() - Step 3: Generate all possible authblock_lines factor combinations
//
void Legal::CreateSpace(model::Engine::Specs arch_specs)
{
  (void) arch_specs; // Suppress unused parameter warning

  std::cout << "Step 3: Creating layout candidate space from authblock_lines factors..." << std::endl;

  unsigned num_storage_levels = layout_.size();
  unsigned num_data_spaces = layout_.at(0).intraline.size();

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

  // If no variable factors found, we have only one candidate (the original layout)
  if (variable_authblock_factors_.empty())
  {
    std::cout << "  No variable authblock_lines factors found. Single layout candidate." << std::endl;
    num_layout_candidates = 1;
    return;
  }

  // Calculate total number of combinations
  num_layout_candidates = 1;
  authblock_factor_ranges_.clear();

  for (const auto& var_factor : variable_authblock_factors_)
  {
    uint32_t max_factor = std::get<3>(var_factor);
    std::vector<uint32_t> divisors = FindDivisors(max_factor);
    authblock_factor_ranges_.push_back(divisors); // Store all divisors of max_factor
    num_layout_candidates *= divisors.size();
  }

  std::cout << "  Total authblock_lines layout candidates: " << num_layout_candidates << std::endl;
  std::cout << "  Variable factors count: " << variable_authblock_factors_.size() << std::endl;
  std::cout << "  Note: Only using divisors of max_factor for each variable factor" << std::endl;
  std::cout << "  ✓ Layout candidate space created successfully" << std::endl;
}

} // namespace layoutspace
