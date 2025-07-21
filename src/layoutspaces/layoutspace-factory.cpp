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

#include "layoutspaces/layoutspace-factory.hpp"
#include <map>
#include <iostream>
#include <cassert>
#include <string>
#include "mapping/loop.hpp"

#define DEBUG 

namespace layoutspace
{

/*
                                                ┌────────────────────────────┐
                                                │ 1. DEFINE MAPPING SPACE    │
                                                │    (all legal loop‑nests)  │
                                                └────────────┬───────────────┘
                                                             │
                                                             ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │ 2. ITERATE: pick next mapping M ∈ mapping‑space          │
                     │    – Identify the spatial loops at every memory level    │
                     │    – requested_parallelism = ∏ extents of those loops    │
                     └────────────┬─────────────────────────────────────────────┘
                                  │
                                  ▼
               ┌───────────────────────────────────────────────────────┐
               │ 3. FOR each 2‑D on‑chip buffer level L                │
               │    – line_cap(L) = words per line (hardware)          │
               └────────────┬──────────────────────────────────────────┘
                            │
                            ▼
               ┌────────────────────────────────────────────────────────┐
               │ 4. DECIDE: how does line_cap(L) compare to             │
               │    requested_parallelism (RP)?                         │
               └────────────┬──────────────────────┬────────────────────┐
                            │                      │                    │
                            ▼                      ▼                    ▼
            ┌─────────────────────┐  ┌─────────────────────┐   ┌─────────────────────┐
            │ 4A. RP == line_cap  │  │ 4B. RP  > line_cap  │   │ 4C. RP  < line_cap  │
            └──────────┬──────────┘  └──────────┬──────────┘   └──────────┬──────────┘
                       │                        │                         │
                       ▼                        ▼                         ▼
   ┌─────────────────────────────┐   ┌─────────────────────────────┐ ┌─────────────────────────────┐
   │ Case 1: Perfect fit.        │   │ Case 2: Line too small.     │ │ Case 3: Line has slack.     │
   │ • If exactly one dim in RP: │   │ • Enumerate partitions of   │ │ • Pack all RP data first.   │
   │   – Enumerate all factor‑   │   │   RP across ⌈RP/line_cap⌉   │ │ • Enumerate temporal‑loop   │
   │     izations of that dim.   │   │   lines (choose subset per  │ │   dimensions that can be    │
   │ • Else (>1 dims):           │   │   line).                    │ │   packed into remaining     │
   │   – Enumerate all flatten‑  │   │ • Continue until every line │ │   slots.                    │
   │     ings of the RP dims     │   │   layout fits in buffer.    │ │ • Continue until buffer‑    │
   │   – (choose which dims map  │   └─────────────────────────────┘ │   size constraint met.      │
   │     to row, order, etc.)    │                                   └─────────────────────────────┘
   └─────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────────────┐
        │ 5. FILTER layouts that violate any constraint:   │
        │    – #lines(layout,L) ≤ #physical_lines(L)       │
        │    – Data required by mapping M is contained     │
        │      within layout rows (no extra stalls).       │
        └────────────┬─────────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────────────┐
        │ 6. EVALUATE each legal (M, layout) pair:         │
        │    – Timeloop cost model → {cycles, energy, …}   │
        │    – Record best‑of‑class metrics or Pareto set  │
        └────────────┬─────────────────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────────────────────┐
        │ 7. ANY mappings left?                           │
        └───────┬──────────────────────────────┬──────────┘
                │Yes                           │No
                ▼                              ▼
         (Return to step 2)         ┌────────────────────────┐
                                    │ 8. OUTPUT optimal      │
                                    │    (mapping, layout)   │
                                    │    configurations      │
                                    └────────────────────────┘
*/

//--------------------------------------------//
//      Parser and LayoutSpace Factory        //
//--------------------------------------------//

LayoutSpace* ParseAndConstruct(config::CompoundConfigNode config,
                               model::Engine::Specs& arch_specs,
                               const Mapping& mapping,
                               layout::Layouts& layout)
{
  // ToDo: @Jianming Need to define a layout constraint.
  LayoutSpace* layoutspace = nullptr;

  std::string layoutspace_template = "Legal";
  config.lookupValue("template", layoutspace_template);

  if (layoutspace_template == "Legal")
  {
    layoutspace = new Legal(arch_specs, mapping, layout, false);
  }
  else
  {
    std::cerr << "ERROR: unsupported layoutspace template: " << layoutspace_template << std::endl;
    exit(-1);
  }

  return layoutspace;
}

//--------------------------------------------//
//        Create Default LayoutSpace          //
//--------------------------------------------//

LayoutSpace* CreateLayoutSpace(const Mapping& mapping,
                               model::Engine::Specs& arch_specs,
                               layout::Layouts& layout,
                              bool skip_init)
{

  return new Legal(arch_specs, mapping, layout, skip_init);
}

//--------------------------------------------//
//    Create Concordant Layout Standalone     //
//--------------------------------------------//

layout::Layouts CreateConcordantLayoutStandalone(const Mapping& mapping, layout::Layouts& layout)
{
  // Copy input layout to layout_local as a new independent variable
  layout::Layouts layout_local = layout;
  
  // Clear authblock nested loops if they're non-empty
  for (unsigned lvl = 0; lvl < layout_local.size(); lvl++)
  {
    for (unsigned ds_idx = 0; ds_idx < layout_local[lvl].authblock_lines.size(); ds_idx++)
    {
      if (!layout_local[lvl].authblock_lines[ds_idx].factors.empty())
      {
        // Clear the authblock factors to make it empty
        layout_local[lvl].authblock_lines[ds_idx].factors.clear();
#ifdef DEBUG
        std::cout << "Cleared authblock_lines factors for level " << lvl << ", dataspace " << ds_idx << std::endl;
#endif
      }
    }
  }

#ifdef DEBUG
  std::cout << "Total number of storage levels: " << mapping.loop_nest.storage_tiling_boundaries.size() << std::endl;
  std::cout << "Total number of layout levels: " << layout_local.size() << std::endl;
  assert(mapping.loop_nest.storage_tiling_boundaries.size() == layout_local.size());
  std::cout << "Total number of data spaces: " << layout_local.at(0).intraline.size() << std::endl;
#endif

  // Build a initialized map that assigns 1 to every dimension ID present in dim_order.
  std::map<std::uint32_t, std::uint32_t> initial_dimid_to_loopend;
  for (char dim_char : layout_local.at(0).dim_order)
  {
    // Convert the char stored in dim_order to a std::string so it can be used
    // as a key into the dimensionToDimID map.
    std::string dim_name(1, dim_char);

    // Look up the dimension ID associated with this name.
    auto dim_id_itr = layout_local.at(0).dimensionToDimID.find(dim_name);
    if (dim_id_itr == layout_local.at(0).dimensionToDimID.end())
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
  unsigned num_data_spaces = layout_local.at(0).intraline.size();
  unsigned inv_storage_level = num_storage_levels;

  // Each storage level vector element starts as a copy of the prototype map.
  std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_interline_dimid_to_loopend(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);
  std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_intraline_dimid_to_loopend(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);

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

  /*
      Step 2: Assign collapsed nested loop to the layout.
  */
  for(unsigned lvl=0; lvl < storage_level_intraline_dimid_to_loopend.size(); lvl++){
    for (unsigned i = 0; i < num_data_spaces; i++){ // iterate over all data spaces
      for(auto & rank: layout_local.at(lvl).intraline.at(i).ranks){ // iterate over all ranks of the data space
        const auto& dim_ids = layout_local.at(lvl).rankToFactorizedDimensionID.at(rank);
        uint32_t total = 0;
        if (dim_ids.size() > 1){
          const auto& coefficient = layout_local.at(lvl).rankToCoefficientValue[rank];
          for (unsigned idx=0; idx < dim_ids.size(); idx++){
            auto dim_value = storage_level_intraline_dimid_to_loopend[lvl][dim_ids[idx]];
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
#ifdef DEBUG
            std::cout << "dim_value=" << dim_value << "--coef[" << idx << "]=" << coefficient[idx] << "; ";
#endif
          }
        }
        else{
          auto dim_value = storage_level_intraline_dimid_to_loopend[lvl][dim_ids[0]];
          total = dim_value;
        }

        layout_local.at(lvl).intraline.at(i).factors.at(rank) = total;
#ifdef DEBUG
        std::cout << "level=" << lvl << " i=" << i << " rank=" << rank << " intraline = " << total << std::endl;
#endif

        total = 0;
        if (dim_ids.size() > 1){
          const auto& coefficient = layout_local.at(lvl).rankToCoefficientValue[rank];
          for (unsigned idx=0; idx < dim_ids.size(); idx++){
            auto dim_value = storage_level_interline_dimid_to_loopend[lvl][dim_ids[idx]];
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
          auto dim_value = storage_level_interline_dimid_to_loopend[lvl][dim_ids[0]];
          total = dim_value;
        }

        layout_local.at(lvl).interline.at(i).factors.at(rank) = total;
      }
    }
  }
  return layout_local;
}

} // namespace layoutspace
