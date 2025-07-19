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

namespace layoutspace
{

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
  CheckBufferCapacityConstraint(arch_specs, mapping);

  // Step 3: Reform layout to be legal
  ReformLayoutToLegal(arch_specs);

  if (!skip_init)
  {
    Init();
  }
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
void Legal::Init()
{
  // Setup all the layout sub-spaces.
  InitDataLayoutSpace();       // Define the place holder for (1) layout constraint and (2) optimal layout candidate

  // Sanity checks.
  for (int i = 0; i < int(layoutspace::Dimension::Num); i++)
  {
    std::cout << "LayoutSpace Dimension [" << layoutspace::Dimension(i)
              << "] Size: " << size_[i] << std::endl;
  }

  // Check for integer overflow in the above multiplications.
  uint128_t de_cumulative_prod = Size();
  for (int i = 0; i < int(layoutspace::Dimension::Num); i++)
  {
    de_cumulative_prod /= size_[i];
  }
  if (de_cumulative_prod != 1)
  {
    std::cerr << "ERROR: overflow detected: layoutspace size appears to be "
              << "greater than 2^128. Please add some layoutspace constraints."
              << std::endl;
    exit(1);
  }
}

//
// InitDataLayoutSpace()
//
void Legal::InitDataLayoutSpace()
{
  // Initialize available data layout patterns
  available_patterns_ = {"linear", "blocked", "interleaved"};
  size_[int(Dimension::Intraline)] = available_patterns_.size();
}

//
// InitPruned()
//
void Legal::InitPruned(uint128_t layout_id)
{
  (void)layout_id; // Suppress unused parameter warning
  // For now, just call regular init
  // In the future, this could initialize a pruned subset based on layout_id
  Init();
}

//
// Split()
//
std::vector<LayoutSpace*> Legal::Split(std::uint64_t num_splits)
{
  std::vector<LayoutSpace*> retval;

  // For now, create identical copies for each split
  // In a more sophisticated implementation, we would divide the space
  for (std::uint64_t i = 0; i < num_splits; i++)
  {
    auto split = new Legal(arch_specs_, mapping_, layout_, true);
    split->split_id_ = i;
    split->num_parent_splits_ = num_splits;
    split->Init();

    splits_.push_back(split);
    retval.push_back(split);
  }

  return retval;
}

//
// ConstructLayout()
//
std::vector<Status> Legal::ConstructLayout(ID layout_id, layout::Layouts* layouts, bool break_on_failure)
{
  (void)break_on_failure; // Suppress unused parameter warning for skeleton implementation
  std::vector<Status> status_per_level;

  // Decode the layout ID into component choices
  auto data_layout_idx = static_cast<size_t>(layout_id[int(Dimension::Intraline)]);
  (void) data_layout_idx; // Suppress unused parameter warning for now
  (void) layouts; // Suppress unused parameter warning for now
  // Create layouts for each storage level
  // @ToDo: iterate layout space and return a layout for evaluation.

  // Return success status
  Status success_status;
  success_status.success = true;
  success_status.fail_reason = "";
  status_per_level.push_back(success_status);

  return status_per_level;
}

//
// CreateConcordantLayout() - Step 1: Create  layout from mapping
//
void Legal::CreateConcordantLayout(const Mapping& mapping)
{
  std::cout << "Step 1: Creating concordant layout from mapping..." << std::endl;
  std::cout << "Total number of storage levels: " << mapping.loop_nest.storage_tiling_boundaries.size() << std::endl;
  std::cout << "Total number of layout levels: " << layout_.size() << std::endl;
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
  unsigned inv_storage_level = num_storage_levels-2; 

  // For DEBUG PURPOSES, we keep the original code below.
  /*
  std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_interline_dimid_to_loopend(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);
  std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_intraline_dimid_to_loopend(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);

  std::string indent = "";
  for (unsigned loop_level = num_loops-1; loop_level != static_cast<unsigned>(-1); loop_level--)
  {
    if (inv_storage_level != static_cast<unsigned>(-1) &&
        mapping.loop_nest.storage_tiling_boundaries.at(inv_storage_level) == loop_level)
    {
      std::cout << "------------------------------------------" << std::endl;
      inv_storage_level--;
    }

    std::cout << indent;
    indent += "  ";

    unsigned cur_storage_level = mapping.loop_nest.storage_tiling_boundaries.size() - inv_storage_level - 2;

    std::cout << "for " << mapping.loop_nest.problem_shape.FlattenedDimensionIDToName.at(mapping.loop_nest.loops.at(loop_level).dimension) << " in [" << mapping.loop_nest.loops.at(loop_level).start << ":" << mapping.loop_nest.loops.at(loop_level).end;
    if (mapping.loop_nest.loops.at(loop_level).residual_end != mapping.loop_nest.loops.at(loop_level).end)
      std::cout << "," << mapping.loop_nest.loops.at(loop_level).residual_end;
    std::cout << ")";
    if (loop::IsSpatial(mapping.loop_nest.loops.at(loop_level).spacetime_dimension))
    {
      storage_level_intraline_dimid_to_loopend[cur_storage_level][mapping.loop_nest.loops.at(loop_level).dimension] = mapping.loop_nest.loops.at(loop_level).end;
    if (loop::IsSpatialX(mapping.loop_nest.loops.at(loop_level).spacetime_dimension))
        std::cout << " (Spatial-X)";
      else
        std::cout << " (Spatial-Y)";
    }else{
      storage_level_interline_dimid_to_loopend.at(cur_storage_level)[mapping.loop_nest.loops.at(loop_level).dimension] = mapping.loop_nest.loops.at(loop_level).end;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
*/
  // Each storage level vector element starts as a copy of the prototype map.
  std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_interline_dimid_to_loopend(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);
  std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_intraline_dimid_to_loopend(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);
  std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_overall_dimval(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);

  for (unsigned loop_level = num_loops-1; loop_level != static_cast<unsigned>(-1); loop_level--)
  {
    if (inv_storage_level != static_cast<unsigned>(-1) &&
        mapping.loop_nest.storage_tiling_boundaries.at(inv_storage_level) == loop_level)
    {
      inv_storage_level--;
    }

    unsigned cur_storage_level = mapping.loop_nest.storage_tiling_boundaries.size() - inv_storage_level - 2;

    if (loop::IsSpatial(mapping.loop_nest.loops.at(loop_level).spacetime_dimension))
    {
      storage_level_intraline_dimid_to_loopend[cur_storage_level][mapping.loop_nest.loops.at(loop_level).dimension] = mapping.loop_nest.loops.at(loop_level).end;
    }else{
      storage_level_interline_dimid_to_loopend.at(cur_storage_level)[mapping.loop_nest.loops.at(loop_level).dimension] = mapping.loop_nest.loops.at(loop_level).end;
    }
  }

  for(unsigned lvl=0; lvl < storage_level_intraline_dimid_to_loopend.size(); lvl++){
    for (unsigned i = 0; i < layout_.at(0).intraline.size(); i++){ // iterate over all data
      for (const auto& kv : storage_level_interline_dimid_to_loopend[lvl])
      {
        storage_level_overall_dimval[lvl][kv.first] = storage_level_intraline_dimid_to_loopend[lvl][kv.first] * storage_level_interline_dimid_to_loopend[lvl][kv.first];
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
    unsigned last_lvl = storage_level_overall_dimval.size() - 1;
    cumulatively_product_dimval[last_lvl] = storage_level_overall_dimval[last_lvl];
    
    // Calculate cumulative product from second-to-last level backwards to first level
    for (int lvl = static_cast<int>(storage_level_overall_dimval.size()) - 2; lvl >= 0; lvl--)
    {
      for (const auto& kv : storage_level_overall_dimval[lvl])
      {
        std::uint32_t dim_id = kv.first;
        std::uint32_t current_value = kv.second;
        
        // Multiply current level value with cumulative product from next level
        if (cumulatively_product_dimval[lvl + 1].find(dim_id) != cumulatively_product_dimval[lvl + 1].end())
        {
          cumulatively_product_dimval[lvl][dim_id] = current_value * cumulatively_product_dimval[lvl + 1][dim_id];
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
  for(unsigned lvl=0; lvl < storage_level_intraline_dimid_to_loopend.size(); lvl++){
    for (unsigned i = 0; i < layout_.at(0).intraline.size(); i++){ // iterate over all data spaces
      for(auto & rank: layout_.at(lvl).intraline.at(i).ranks){ // iterate over all ranks of the data space
        const auto& dim_ids = layout_.at(lvl).rankToFactorizedDimensionID.at(rank);
        uint32_t total = 0;
        if (dim_ids.size() > 1){
          const auto& coefficient = layout_.at(lvl).rankToCoefficientValue[rank];
          for (unsigned idx=0; idx < dim_ids.size(); idx++){ 
            auto dim_value = storage_level_intraline_dimid_to_loopend[lvl][dim_ids[idx]];
            if (idx == dim_ids.size()-1)
              total +=  dim_value*coefficient[idx] - 1;
            else
              total +=  dim_value*coefficient[idx];
#ifdef DEBUG_CONCORDANT_LAYOUT
            std::cout << "dim_value=" << dim_value << "--coef[" << idx << "]=" << coefficient[idx] << "; ";
#endif
          }
        }
        else{
          auto dim_value = storage_level_intraline_dimid_to_loopend[lvl][dim_ids[0]];
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
            auto dim_value = storage_level_interline_dimid_to_loopend[lvl][dim_ids[idx]];
            if (idx == dim_ids.size()-1)
              total +=  dim_value*coefficient[idx] - 1;
            else
              total +=  dim_value*coefficient[idx];
          }
        }
        else{
          auto dim_value = storage_level_interline_dimid_to_loopend[lvl][dim_ids[0]];
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
#endif

  std::vector<std::vector<uint32_t>> tensor_size;
  tensor_size.resize(num_storage_levels, std::vector<uint32_t>(num_data_spaces, 0));

  // first level: number storage levels
  // second level: number data spaces
  // third level: size of tensor
  for (unsigned lvl=0; lvl < num_storage_levels; lvl++){
    for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){ 
      uint32_t dataspace_size_cur_lvl = 0;
      for (auto & rank: layout_.at(lvl).intraline.at(ds_idx).ranks){
        auto intraline_rank_value = layout_.at(lvl).intraline.at(ds_idx).factors[rank];
        auto interline_rank_value = layout_.at(lvl).interline.at(ds_idx).factors[rank];
        dataspace_size_cur_lvl += intraline_rank_value * interline_rank_value;
      }
      tensor_size[lvl][ds_idx] = dataspace_size_cur_lvl;
    }
  }

  // Print out the tensor size
  for (unsigned lvl=0; lvl < tensor_size.size(); lvl++){
    std::cout << "Storage level " << lvl << " tensor size: ";
    for (unsigned ds_idx = 0; ds_idx < tensor_size[lvl].size(); ds_idx++){
      std::cout << tensor_size[lvl][ds_idx] << " ";
    }
    std::cout << std::endl;
  }
}


//
// ReformLayoutToLegal() - Step 3: Reform layout to be legal with mapping
//
void Legal::ReformLayoutToLegal(model::Engine::Specs arch_specs)
{
  std::cout << "Step 3: Reforming layout to be legal..." << std::endl;

  // Calculate requested parallelism (RP) as product of spatial loop extents
  uint64_t requested_parallelism = 1;
  for (const auto& spatial_loop : concordant_spatial_loops_)
  {
    requested_parallelism *= spatial_loop.second;
  }

  std::cout << "  Requested Parallelism (RP) = " << requested_parallelism << std::endl;

  // Define buffer line capacity (this would typically come from architecture specs)
  // For now, using a representative value - in real implementation this should come from arch_specs_
  uint64_t line_cap = 64; // Example line capacity
  std::cout << "  Buffer Line Capacity = " << line_cap << std::endl;

  // Step 2.1: Check if all requested data fits in on-chip buffer
  if (!CheckBufferCapacityConstraint(arch_specs, mapping_))
  {
    std::cout << "  ERROR: Requested data does not fit in on-chip buffer. Legal space is empty." << std::endl;
    legal_space_empty_ = true;
    return;
  }

  // Step 2.2: Handle different cases based on RP vs line_cap
  if (requested_parallelism == line_cap)
  {
    std::cout << "  RP == line_cap: No additional tiling needed" << std::endl;
    // Do nothing - current layout is already legal
  }
  else if (requested_parallelism > line_cap)
  {
    std::cout << "  RP > line_cap: Need further tiling and loop movement" << std::endl;
    HandleOverParallelism(requested_parallelism, line_cap);
  }
  else // requested_parallelism < line_cap
  {
    std::cout << "  RP < line_cap: Can pack more data and tile temporal loops" << std::endl;
    HandleUnderParallelism(requested_parallelism, line_cap);
  }

  legal_space_empty_ = false;
  std::cout << "  Layout reform completed successfully" << std::endl;
}

//
// CheckBufferCapacityConstraint() - Check if data fits in buffer
//
bool Legal::CheckBufferCapacityConstraint(model::Engine::Specs arch_specs, const Mapping& mapping)
{
  std::cout << "  Checking buffer capacity constraints..." << std::endl;
  
  unsigned num_storage_levels = arch_specs.topology.NumStorageLevels();
  unsigned num_data_spaces = layout_.at(0).intraline.size();
  
  // Validate that we have dimension values for each storage level
  if (cumulatively_product_dimval.size() != num_storage_levels)
  {
    std::cout << "    ERROR: Mismatch between storage levels (" << num_storage_levels 
              << ") and dimension value levels (" << cumulatively_product_dimval.size() << ")" << std::endl;
    return false;
  }
  
  // Get bypass information from mapping using direct bitset access
  std::cout << "  Analyzing data space bypass configuration:" << std::endl;
  for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++)
  {
    auto storage_level_specs = arch_specs.topology.GetStorageLevel(storage_level);
    std::cout << "    Level " << storage_level << " (" << storage_level_specs->name.Get() << "): ";
    
    std::vector<std::string> kept_data_spaces, bypassed_data_spaces;
    
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
  }
  
  // Iterate through each storage level
  for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++)
  {
    auto storage_level_specs = arch_specs.topology.GetStorageLevel(storage_level);
    
    // Extract memory specifications
    std::uint64_t total_capacity = 0;
    std::uint64_t block_size = 0;
    std::uint64_t line_capacity = 0;
    std::uint64_t word_bits = 0;
    double read_bandwidth = 0.0;
    double write_bandwidth = 0.0;
    
    // Extract word_bits
    if (storage_level_specs->word_bits.IsSpecified())
    {
      word_bits = storage_level_specs->word_bits.Get();
    }
    
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
                << " (" << storage_level_specs->name.Get() << ") has unspecified size" << std::endl;
      continue;
    }
    
    // Line capacity is the number of elements that can be accessed in parallel
    // Use the maximum of read_bandwidth and write_bandwidth, or block_size as fallback
    line_capacity = static_cast<std::uint64_t>(std::max(read_bandwidth, write_bandwidth));
    if (line_capacity == 0)
    {
      line_capacity = block_size;  // Fallback to block_size
    }
    
    // Calculate memory depth from total capacity and block size
    std::uint64_t memory_depth = (block_size > 0) ? (total_capacity / block_size) : 0;
    
    std::cout << "    Storage Level " << storage_level 
              << " (" << storage_level_specs->name.Get() << "):" << std::endl;
    std::cout << "      Memory depth: " << memory_depth << " lines" << std::endl;
    std::cout << "      Block size: " << block_size << " elements per line" << std::endl;
    std::cout << "      Total capacity: " << total_capacity << " elements" << std::endl;
    std::cout << "      Line capacity (bandwidth): " << line_capacity << " elements/cycle" << std::endl;
    std::cout << "      Word bits: " << word_bits << " bits" << std::endl;
    std::cout << "      Read bandwidth: " << read_bandwidth << " elements/cycle" << std::endl;
    std::cout << "      Write bandwidth: " << write_bandwidth << " elements/cycle" << std::endl;
    
    // Get cumulative dimension values for this storage level
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
          
          // Calculate rank size using cumulative dimension values
          std::uint64_t rank_dimension_product = 1;
          for (auto dim_id : dim_ids)
          {
            if (level_dimval.find(dim_id) != level_dimval.end())
            {
              rank_dimension_product *= level_dimval.at(dim_id);
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
      
      std::cout << "      Data space " << ds_idx;
      if (ds_idx < layout_.at(storage_level).data_space.size())
      {
        std::cout << " (" << layout_.at(storage_level).data_space[ds_idx] << ")";
      }
      std::cout << (is_kept ? " [KEPT]" : " [BYPASSED]") << ":" << std::endl;
      std::cout << "        Total data size: " << dataspace_total_size << " elements" << std::endl;
      std::cout << "        Parallel accesses: " << dataspace_parallel_size << " elements" << std::endl;
    }
    
    std::cout << "      TOTAL data size across all spaces: " << total_data_size << " elements" << std::endl;
    std::cout << "      TOTAL parallel accesses: " << total_parallel_accesses << " elements" << std::endl;
    
    // Check capacity constraint: total data size should not exceed buffer capacity
    if (total_data_size > total_capacity)
    {
      std::cout << "    ERROR: Total data size (" << total_data_size 
                << ") exceeds memory capacity (" << total_capacity << ")" << std::endl;
      return false;
    }
    
    // Check bandwidth constraint: parallel accesses must fit in line capacity
    if (total_parallel_accesses > line_capacity)
    {
      std::cout << "    ERROR: Total parallel accesses (" << total_parallel_accesses 
                << ") exceed line capacity (" << line_capacity << ")" << std::endl;
      return false;
    }
    
    std::cout << "    ✓ Storage level " << storage_level << " constraints satisfied" << std::endl;
  }
  
  std::cout << "  ✓ All buffer capacity constraints satisfied" << std::endl;
  return true;
}

//
// HandleOverParallelism() - Handle case where RP > line_cap
//
void Legal::HandleOverParallelism(uint64_t requested_parallelism, uint64_t line_cap)
{
  std::cout << "    Implementing further intraline tiling..." << std::endl;

  // Calculate tiling factor needed
  uint64_t tiling_factor = (requested_parallelism + line_cap - 1) / line_cap; // Ceiling division
  std::cout << "    Tiling factor needed: " << tiling_factor << std::endl;

  // Create new design space for:
  // 1. Which spatial loops to tile further
  // 2. How to move some loops from intraline to interline

  // For demonstration, tile the largest spatial loop
  if (!concordant_spatial_loops_.empty())
  {
    auto& largest_loop = *std::max_element(concordant_spatial_loops_.begin(),
                                          concordant_spatial_loops_.end(),
                                          [](const auto& a, const auto& b) { return a.second < b.second; });

    std::cout << "    Tiling spatial loop " << largest_loop.first
              << " (size " << largest_loop.second << ")" << std::endl;

    // Split the loop: part stays intraline, part becomes interline
    uint64_t intraline_portion = largest_loop.second / tiling_factor;
    uint64_t interline_portion = tiling_factor;

    std::cout << "    Split: intraline=" << intraline_portion
              << ", interline=" << interline_portion << std::endl;
  }
}

//
// HandleUnderParallelism() - Handle case where RP < line_cap
//
void Legal::HandleUnderParallelism(uint64_t requested_parallelism, uint64_t line_cap)
{
  std::cout << "    Packing spatial data and tiling temporal loops..." << std::endl;

  uint64_t remaining_capacity = line_cap - requested_parallelism;
  std::cout << "    Remaining line capacity: " << remaining_capacity << std::endl;

  // Enumerate temporal loops that can be packed into remaining slots
  for (const auto& temporal_loop : concordant_temporal_loops_)
  {
    if (remaining_capacity >= temporal_loop.second)
    {
      std::cout << "    Can pack temporal loop " << temporal_loop.first
                << " (size " << temporal_loop.second << ") into remaining slots" << std::endl;
      remaining_capacity -= temporal_loop.second;
    }
    else if (remaining_capacity > 1)
    {
      // Tile the temporal loop to fit remaining capacity
      uint64_t tile_size = remaining_capacity;
      std::cout << "    Tiling temporal loop " << temporal_loop.first
                << " with tile size " << tile_size << std::endl;
      remaining_capacity = 0;
      break;
    }
  }

  std::cout << "    Final remaining capacity: " << remaining_capacity << std::endl;
}

} // namespace layoutspace