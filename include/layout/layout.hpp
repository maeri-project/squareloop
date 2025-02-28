#pragma once

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <yaml-cpp/yaml.h>
#include "mapping/nest.hpp"

//------------------------------------------------------------------------------
// Layout structure (target-level information)
//------------------------------------------------------------------------------


namespace layout
{

struct Layout {
  std::string target;
  loop::Nest interline;   // single interline nest
  loop::Nest intraline;   // single intraline nest
  int num_read_ports = 1;   // from the interline specification (default 1)
  int num_write_ports = 1;  // from the interline specification (default 1)
  std::vector<int> max_dim_perline; // computed from the intraline nest
  bool initialize = false;  // true if external YAML config provides layout
  std::vector<char> factor_order; // records the factor letters in sorted order
};

typedef std::vector<Layout> Layouts;

//------------------------------------------------------------------------------
// Helper: parseOrderMapping()
// Parses a mapping string (e.g., "C:0, M:1, R:2, S:3, N:4, P:5, Q:6")
// into an unordered_map from char to int.
//------------------------------------------------------------------------------
std::map<std::string, unsigned> parseOrderMapping(const std::string &mappingString);

//------------------------------------------------------------------------------
// ParseAndConstruct()
// This function uses the compound-config library to read a configuration that has a top-level
// "layout" array. Each entry must contain:
//   - target (string)
//   - type (string): either "interline" or "intraline"
//   - factors (string): e.g., "R=3 S=3 P=7 Q=7 C=3 M=1 N=1"
//   - permutation (string): e.g., "SR CQP MN"
// For interline entries, optional fields "num_read_ports" and "num_write_ports" are parsed (defaulting to 1).
//
// For each unique target, a Layout is created holding one interline nest and one intraline nest.
// If a nest is missing, a default nest with all factors set to 1 is created.
// Also, the extra vector factor_order is set from the externally provided order mapping.
// Finally, max_dim_perline is computed from the intraline nest.
  
std::vector<Layout> ParseAndConstruct(config::CompoundConfigNode layoutArray,
                                                 problem::Workload& workload,
      std::map<std::string, std::pair<uint64_t,uint64_t>>& externalPortMapping);

//------------------------------------------------------------------------------
// Helper function to print a Nest's loop order.
//------------------------------------------------------------------------------
void PrintNestLoopOrder(const loop::Nest &nest, const std::vector<char>& factorOrder);

} // namespace layout