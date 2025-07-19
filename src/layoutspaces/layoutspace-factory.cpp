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
    layoutspace = new Legal(arch_specs, mapping, layout);
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
                               layout::Layouts& layout)
{

  return new Legal(arch_specs, mapping, layout, false);
}

} // namespace layoutspace
