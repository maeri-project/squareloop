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

#include "layoutspaces/layoutspace-base.hpp"

//--------------------------------------------//
//           Layoutspace Dimensions           //
//--------------------------------------------//

namespace layoutspace
{

std::ostream& operator << (std::ostream& out, Dimension d)
{
  switch (d)
  {
  case Dimension::Intraline:
    out << "Intraline";
    break;
  case Dimension::Num:
    out << "Num";
    break;
  default:
    out << "Unknown";
    break;
  }
  return out;
}

//--------------------------------------------//
//            LayoutSpace Base Class          //
//--------------------------------------------//

std::vector<Status> LayoutSpace::ConstructLayout(ID layout_id, layout::Layouts* layouts, Mapping mapping, bool break_on_failure)
{
  // Default implementation - should be overridden by derived classes
  (void)layout_id;
  (void)layouts; 
  (void)mapping;
  (void)break_on_failure;
  
  Status error_status;
  error_status.success = false;
  error_status.fail_reason = "ConstructLayout not implemented in base class";
  return {error_status};
}

std::vector<Status> LayoutSpace::ConstructLayout(uint64_t layout_splitting_id, uint64_t layout_auth_id, uint64_t layout_packing_id, layout::Layouts* layouts, Mapping mapping, bool break_on_failure)
{
  // Default implementation - should be overridden by derived classes
  (void)layout_splitting_id;
  (void)layout_auth_id;
  (void)layout_packing_id;
  (void)layouts;
  (void)mapping;
  (void)break_on_failure;
  
  Status error_status;
  error_status.success = false;
  error_status.fail_reason = "ConstructLayout not implemented in base class";
  return {error_status};
}

} // namespace layoutspace 