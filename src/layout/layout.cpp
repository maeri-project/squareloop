#include "layout/layout.hpp"


namespace layout
{

//------------------------------------------------------------------------------
// Helper: parseOrderMapping()
// Parses a mapping string (e.g., "C:0, M:1, R:2, S:3, N:4, P:5, Q:6")
// into an unordered_map from char to int.
//------------------------------------------------------------------------------
std::map<std::string, unsigned> ParseOrderMapping(const std::string &mappingString) {
  std::map<std::string, unsigned> orderMapping;
  std::istringstream iss(mappingString);
  std::string token;
  while (std::getline(iss, token, ',')) {
    token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
    if (token.empty()) continue;
    size_t pos = token.find(':');
    if (pos != std::string::npos)
      orderMapping[token.substr(0,1)] = static_cast<unsigned>(std::stoi(token.substr(pos+1)));
  }
  return orderMapping;
}

//------------------------------------------------------------------------------
// ParseAndConstruct()
// This function uses the compound-config library to read a configuration that has a top-level
// "layout" array. Each entry must contain:
//   - target (string)
//   - type (string): either "interline" or "intraline"
//   - factors (string): e.g., "R=3 S=3 P=7 Q=7 C=3 M=1 N=1"
//   - permutation (string): The permutation string is processed by removing whitespace and then reversed,
// so that left-to-right order is interpreted as outer-most to inner-most. 
//  e.g., "SR CQP MN" will become "SRCQPMN" with all spaces being ignored
// 
// For interline entries, optional fields "num_read_ports" and "num_write_ports" are parsed.
// For each unique target, a Layout is created holding one interline nest and one intraline nest.
// If a nest is missing, a default nest (with all factors set to 1) is created.
// Also, factor_order is recorded (from the external mapping) and max_dim_perline is computed from the intraline nest.
// (as defined by the external order mapping) the product of the interline and
// intraline factors is >= the corresponding bound in dimension_bound.
// Finally, the function returns a std::vector<Layout> containing one Layout per unique target.
std::vector<Layout> ParseAndConstruct(config::CompoundConfigNode layoutArray,
  problem::Workload& workload,
std::map<std::string, std::pair<uint64_t,uint64_t>>& externalPortMapping){
  const std::map<std::string, unsigned> externalMapping = workload.GetShape()->FactorizedDimensionNameToID;
  const std::vector<std::int32_t> dimension_bound = workload.GetFactorizedBounds().GetCoordinates();
  std::unordered_map<std::string, Layout> layoutMap;
  int numEntries = layoutArray.getLength();
  for (int i = 0; i < numEntries; i++) {
    config::CompoundConfigNode entry = layoutArray[i];
    std::string targetName, layoutType, factorsStr, permutationStr;
    if (!entry.lookupValue("target", targetName) ||
        !entry.lookupValue("type", layoutType) ||
        !entry.lookupValue("factors", factorsStr) ||
        !entry.lookupValue("permutation", permutationStr))
      continue;
    // Build mapping from factor letter to loop bound.
    std::unordered_map<char,int> factorValues;
    {
      std::istringstream facStream(factorsStr);
      std::string token;
      while (facStream >> token) {
        size_t pos = token.find('=');
        if (pos != std::string::npos)
          factorValues[token[0]] = std::stoi(token.substr(pos+1));
      }
    }
    // Remove whitespace from permutation string.
    std::string cleanPermutation;
    for (char c : permutationStr)
      if (!std::isspace(static_cast<unsigned char>(c)))
        cleanPermutation.push_back(c);
    // Modification: process permutation left-to-right as innermost to outermost,
    // so reverse the cleanPermutation.
    std::reverse(cleanPermutation.begin(), cleanPermutation.end());
    loop::Nest parsedNest;
    loop::Nest::SkewDescriptor skewDescriptor;
    for (char f : cleanPermutation) {
      loop::Nest::SkewDescriptor::Term term;
      term.constant = (factorValues.find(f) != factorValues.end()) ? factorValues[f] : 1;
      std::string factorKey(1, f);
      if (externalMapping.find(factorKey) != externalMapping.end()) {
        term.variable.dimension = externalMapping.at(factorKey);
        term.bound.dimension = externalMapping.at(factorKey);
      } else {
        term.variable.dimension = f - 'A';
        term.bound.dimension = f - 'A';
      }
      if (layoutType == "intraline") {
        term.variable.is_spatial = (term.constant > 1);
        term.bound.is_spatial = (term.constant > 1);
      } else {
        term.variable.is_spatial = false;
        term.bound.is_spatial = false;
      }
      skewDescriptor.terms.push_back(term);
    }
    parsedNest.skew_descriptors.insert({0, skewDescriptor});
    if (layoutMap.find(targetName) == layoutMap.end()) {
      Layout newLayout;
      newLayout.target = targetName;
      newLayout.initialize = true;
      layoutMap[targetName] = newLayout;
    }
    if (layoutType == "interline") {
      if (layoutMap[targetName].interline.skew_descriptors.empty()) {
        layoutMap[targetName].interline = parsedNest;
        if (externalPortMapping.find(targetName) != externalPortMapping.end()) {
          layoutMap[targetName].num_read_ports = static_cast<int>(externalPortMapping.at(targetName).first);
          layoutMap[targetName].num_write_ports = static_cast<int>(externalPortMapping.at(targetName).second);
        } else {
          layoutMap[targetName].num_read_ports = 1;
          layoutMap[targetName].num_write_ports = 1;
        }
      }
    } else if (layoutType == "intraline") {
      if (layoutMap[targetName].intraline.skew_descriptors.empty())
        layoutMap[targetName].intraline = parsedNest;
    } else {
      if (layoutMap[targetName].interline.skew_descriptors.empty()) {
        layoutMap[targetName].interline = parsedNest;
        if (externalPortMapping.find(targetName) != externalPortMapping.end()) {
          layoutMap[targetName].num_read_ports = static_cast<int>(externalPortMapping.at(targetName).first);
          layoutMap[targetName].num_write_ports = static_cast<int>(externalPortMapping.at(targetName).second);
        } else {
          layoutMap[targetName].num_read_ports = 1;
          layoutMap[targetName].num_write_ports = 1;
        }
      }
    }
  } // end for each layout entry

  // Build sorted factor order from externalMapping.
  std::vector<std::pair<unsigned, std::string>> sortedFactorPairs;
  for (const auto &p : externalMapping)
    sortedFactorPairs.push_back({p.second, p.first});
  std::sort(sortedFactorPairs.begin(), sortedFactorPairs.end(),
            [](auto a, auto b){ return a.first < b.first; });
  std::vector<char> globalFactorOrder;
  for (const auto &pair : sortedFactorPairs)
    globalFactorOrder.push_back(pair.second[0]);
  if (dimension_bound.size() != globalFactorOrder.size()) {
    std::cerr << "Error: dimension_bound size (" << dimension_bound.size() 
              << ") does not match external mapping size (" << globalFactorOrder.size() << ").\n";
    return {};
  }
  // Create default nests if missing and assign factor_order.
  for (auto &entry : layoutMap) {
    Layout &l = entry.second;
    if (l.intraline.skew_descriptors.empty()) {
      loop::Nest defaultIntraline;
      loop::Nest::SkewDescriptor defaultSkew;
      for (const auto &pf : sortedFactorPairs) {
        loop::Nest::SkewDescriptor::Term term;
        term.constant = 1;
        term.variable.dimension = pf.first;
        term.bound.dimension = pf.first;
        defaultSkew.terms.push_back(term);
      }
      defaultIntraline.skew_descriptors.insert({0, defaultSkew});
      l.intraline = defaultIntraline;
    }
    if (l.interline.skew_descriptors.empty()) {
      loop::Nest defaultInterline;
      loop::Nest::SkewDescriptor defaultSkew;
      for (const auto &pf : sortedFactorPairs) {
        loop::Nest::SkewDescriptor::Term term;
        term.constant = 1;
        term.variable.dimension = pf.first;
        term.bound.dimension = pf.first;
        defaultSkew.terms.push_back(term);
      }
      defaultInterline.skew_descriptors.insert({0, defaultSkew});
      l.interline = defaultInterline;
    }
    l.factor_order = globalFactorOrder;
    // Compute max_dim_perline as product of interline and intraline factors.
    std::vector<int> computedMaxDims(globalFactorOrder.size(), 1);
    const auto &interlineTerms = l.interline.skew_descriptors.at(0).terms;
    const auto &intralineTerms = l.intraline.skew_descriptors.at(0).terms;
    for (size_t i = 0; i < sortedFactorPairs.size(); i++) {
      unsigned expectedOrder = sortedFactorPairs[i].first;
      int interFactor = 1, intraFactor = 1;
      for (const auto &term : interlineTerms) {
        if (term.variable.dimension == expectedOrder) { interFactor = term.constant; break; }
      }
      for (const auto &term : intralineTerms) {
        if (term.variable.dimension == expectedOrder) { intraFactor = term.constant; break; }
      }
      int product = interFactor * intraFactor;
      if (product < static_cast<int>(dimension_bound[i])) {
        std::cerr << "Warning: For target " << entry.first << ", the product of interline ("
                  << interFactor << ") and intraline (" << intraFactor 
                  << ") factors for dimension " << globalFactorOrder[i]
                  << " is " << product << ", below the required bound " << dimension_bound[i] << ".\n";
      }
      computedMaxDims[i] = product;
    }
    l.max_dim_perline = computedMaxDims;
  }
  // Convert layoutMap into vector.
  std::vector<Layout> layoutVector;
  for (const auto &p : layoutMap)
    layoutVector.push_back(p.second);
  return layoutVector;
}


//------------------------------------------------------------------------------
// Helper function to print a Nest's loop order.
//------------------------------------------------------------------------------
void PrintNestLoopOrder(const loop::Nest &nest, const std::vector<char>& factorOrder) {
  for (const auto &descPair : nest.skew_descriptors) {
    for (const auto &term : descPair.second.terms) {
      int ord = term.variable.dimension;
      char f = (ord < static_cast<int>(factorOrder.size())) ? factorOrder[ord] : '?';
      std::cout << "    iter->dimension=" << ord << "-" << f
                << " in [0, " << term.constant << ", 1) iter->residual_end=" << term.constant << "\n";
    }
  }
}

} // namespace layout