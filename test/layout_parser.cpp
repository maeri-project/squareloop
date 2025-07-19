#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

// Include the compound-config API using the new header path.
#include "compound-config/compound-config.hpp"  // Definitions in namespace config

//------------------------------------------------------------------------------
// Dummy definitions for problem::Shape
//------------------------------------------------------------------------------
namespace problem {
  struct Shape {
    typedef unsigned FlattenedDimensionID;
    static const int NumFlattenedDimensions = 100;
  };
  const Shape* GetShape() {
    static Shape shape;
    return &shape;
  }
}

//------------------------------------------------------------------------------
// Nest class in namespace loop (with dim_char removed)
//------------------------------------------------------------------------------
namespace loop {

class Nest {
 public:
  struct SkewDescriptor {
    struct Term {
      struct DimSpec {
        problem::Shape::FlattenedDimensionID dimension = problem::GetShape()->NumFlattenedDimensions;
        bool is_spatial = false;
      };
      int constant = 1;
      DimSpec variable;
      DimSpec bound;
    };
    std::vector<Term> terms;
    int modulo = 0;
  };
  // Use an unordered_map keyed by unsigned.
  std::unordered_map<unsigned, SkewDescriptor> skew_descriptors;
};

} // namespace loop

//------------------------------------------------------------------------------
// Layout structure (target-level configuration)
//------------------------------------------------------------------------------
struct Layout {
  std::string target;           // e.g., "MainMemory"
  loop::Nest interline;         // interline nest
  loop::Nest intraline;         // intraline nest
  int num_read_ports = 1;       // from interline entry (default 1)
  int num_write_ports = 1;      // from interline entry (default 1)
  std::vector<int> max_dim_perline; // computed from intraline nest
  bool initialize = false;      // true if external YAML config provided layout
  std::vector<char> factor_order;   // sorted order of factor letters (e.g., {'C','M','R','S','N','P','Q'})
};

//------------------------------------------------------------------------------
// ParseOrderMapping()
// Converts a string like "C:0, M:1, R:2, S:3, N:4, P:5, Q:6" into a std::map<string, unsigned>
std::map<std::string, unsigned> ParseOrderMapping(const std::string &mappingString) {
  std::map<std::string, unsigned> orderMapping;
  std::istringstream iss(mappingString);
  std::string token;
  while (std::getline(iss, token, ',')) {
    token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
    if (token.empty())
      continue;
    size_t pos = token.find(':');
    if (pos != std::string::npos)
      orderMapping[token.substr(0,1)] = static_cast<unsigned>(std::stoi(token.substr(pos+1)));
  }
  return orderMapping;
}

//------------------------------------------------------------------------------
// ParseAndConstruct()
// Now takes an additional parameter "dimension_bound" (vector<int32_t>).
// The permutation string is processed by simply removing whitespace,
// and its letters (left-to-right) are interpreted as outer-most to inner-most.
// After constructing each Layout, the function checks that for each dimension
// (as defined by the external order mapping) the product of the interline and
// intraline factors is >= the corresponding bound in dimension_bound.
std::vector<Layout> ParseAndConstruct(config::CompoundConfig* cc,
                                      const std::map<std::string, unsigned>& externalMapping,
                                      const std::vector<std::int32_t>& dimension_bound) {
  // Internal grouping by target.
  std::unordered_map<std::string, Layout> layoutMap;
  config::CompoundConfigNode rootNode = cc->getRoot();
  if (!rootNode.exists("layout")) {
    std::cerr << "No 'layout' node in configuration.\n";
    return {};
  }
  config::CompoundConfigNode layoutArray = rootNode.lookup("layout");
  int numEntries = layoutArray.getLength();
  for (int i = 0; i < numEntries; i++) {
    config::CompoundConfigNode entry = layoutArray[i];
    std::string targetName, layoutType, factorsStr, permutationStr;
    if (!entry.lookupValue("target", targetName) ||
        !entry.lookupValue("type", layoutType) ||
        !entry.lookupValue("factors", factorsStr) ||
        !entry.lookupValue("permutation", permutationStr))
      continue;
    int nr = 1, nw = 1;
    if (layoutType == "interline") {
      entry.lookupValue("num_read_ports", nr);
      entry.lookupValue("num_write_ports", nw);
    }
    // Build a mapping from factor letter to its loop bound.
    std::unordered_map<char,int> factorValues;
    {
      std::istringstream facStream(factorsStr);
      std::string token;
      while (facStream >> token) {
        size_t eqPos = token.find('=');
        if (eqPos != std::string::npos)
          factorValues[token[0]] = std::stoi(token.substr(eqPos+1));
      }
    }
    // Remove all whitespace from permutation string.
    std::string cleanPermutation;
    for (char c : permutationStr)
      if (!std::isspace(static_cast<unsigned char>(c)))
        cleanPermutation.push_back(c);
    // (Interpret cleanPermutation left-to-right as outer-most to inner-most.)
    loop::Nest parsedNest;
    loop::Nest::SkewDescriptor skewDescriptor;
    for (char factorChar : cleanPermutation) {
      loop::Nest::SkewDescriptor::Term term;
      term.constant = (factorValues.find(factorChar) != factorValues.end()) ? factorValues[factorChar] : 1;
      // Lookup the external mapping: convert factorChar to string.
      std::string factorKey(1, factorChar);
      if (externalMapping.find(factorKey) != externalMapping.end()) {
        term.variable.dimension = externalMapping.at(factorKey);
        term.bound.dimension = externalMapping.at(factorKey);
      } else {
        term.variable.dimension = factorChar - 'A';
        term.bound.dimension = factorChar - 'A';
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
    // Group by target.
    if (layoutMap.find(targetName) == layoutMap.end()) {
      Layout newLayout;
      newLayout.target = targetName;
      newLayout.initialize = true;
      layoutMap[targetName] = newLayout;
    }
    if (layoutType == "interline") {
      if (layoutMap[targetName].interline.skew_descriptors.empty()) {
        layoutMap[targetName].interline = parsedNest;
        layoutMap[targetName].num_read_ports = nr;
        layoutMap[targetName].num_write_ports = nw;
      }
    } else if (layoutType == "intraline") {
      if (layoutMap[targetName].intraline.skew_descriptors.empty())
        layoutMap[targetName].intraline = parsedNest;
    } else {
      if (layoutMap[targetName].interline.skew_descriptors.empty()) {
        layoutMap[targetName].interline = parsedNest;
        layoutMap[targetName].num_read_ports = nr;
        layoutMap[targetName].num_write_ports = nw;
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
  // Check that dimension_bound has the same size.
  if (dimension_bound.size() != globalFactorOrder.size()) {
    std::cerr << "Error: dimension_bound size (" << dimension_bound.size() 
              << ") does not match external mapping size (" << globalFactorOrder.size() << ").\n";
    return {};
  }
  // Create default nests if missing and assign factor_order.
  for (auto &entry : layoutMap) {
    Layout &layoutConfig = entry.second;
    if (layoutConfig.intraline.skew_descriptors.empty()) {
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
      layoutConfig.intraline = defaultIntraline;
    }
    if (layoutConfig.interline.skew_descriptors.empty()) {
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
      layoutConfig.interline = defaultInterline;
    }
    layoutConfig.factor_order = globalFactorOrder;
    // Compute max_dim_perline as the product of interline and intraline factors.
    std::vector<int> computedMaxDims(globalFactorOrder.size(), 1);
    // For each dimension, search for the term in interline and intraline.
    // We assume each nest's SkewDescriptor is stored with key 0.
    const auto &interlineTerms = layoutConfig.interline.skew_descriptors.at(0).terms;
    const auto &intralineTerms = layoutConfig.intraline.skew_descriptors.at(0).terms;
    for (size_t i = 0; i < sortedFactorPairs.size(); i++) {
      unsigned expectedOrder = sortedFactorPairs[i].first;
      int interFactor = 1, intraFactor = 1;
      for (const auto &term : interlineTerms) {
        if (term.variable.dimension == expectedOrder) {
          interFactor = term.constant;
          break;
        }
      }
      for (const auto &term : intralineTerms) {
        if (term.variable.dimension == expectedOrder) {
          intraFactor = term.constant;
          break;
        }
      }
      int product = interFactor * intraFactor;
      if (product < static_cast<int>(dimension_bound[i])) {
        std::cerr << "Warning: For target " << entry.first << ", the product of interline (" 
                  << interFactor << ") and intraline (" << intraFactor 
                  << ") factors for dimension " << globalFactorOrder[i]
                  << " is " << product << ", which is below the required bound " << dimension_bound[i] << ".\n";
      }
      computedMaxDims[i] = product;
    }
    layoutConfig.max_dim_perline = computedMaxDims;
  }
  // Convert layoutMap to vector.
  std::vector<Layout> layoutVector;
  for (const auto &l : layoutMap)
    layoutVector.push_back(l.second);
  return layoutVector;
}

//------------------------------------------------------------------------------
// PrintNestLoopOrder()
// Iterates over the unordered_map of skew descriptors and prints each term.
void PrintNestLoopOrder(const loop::Nest &nest, const std::vector<char>& factorOrder) {
  for (const auto &descPair : nest.skew_descriptors) {
    for (const auto &term : descPair.second.terms) {
      int ord = term.variable.dimension;
      char factorLetter = (ord < static_cast<int>(factorOrder.size())) ? factorOrder[ord] : '?';
      std::cout << "    iter->dimension=" << ord << "-" << factorLetter
                << " in [0, " << term.constant << ", 1) iter->residual_end=" << term.constant << "\n";
    }
  }
}

//------------------------------------------------------------------------------
// main()
//------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config file>\n";
    return 1;
  }
  // Define the external order mapping using the new data structure.
  typedef unsigned FactorizedDimensionID;
  std::map<std::string, FactorizedDimensionID> orderMappingInput = {
    {"C", 0}, {"M", 1}, {"R", 2}, {"S", 3}, {"N", 4}, {"P", 5}, {"Q", 6}
  };
  // Define the overall lower limits for each dimension (in the order of orderMappingInput).
  std::vector<std::int32_t> dimension_bound = {3, 1, 3, 3, 1, 7, 7};

  config::CompoundConfig* cc = new config::CompoundConfig(argv[1]);
  std::vector<Layout> layouts = ParseAndConstruct(cc, orderMappingInput, dimension_bound);
  for (const auto &layoutConfig : layouts) {
    std::cout << "Target: " << layoutConfig.target << "\n"
              << "  num_read_ports: " << layoutConfig.num_read_ports
              << ", num_write_ports: " << layoutConfig.num_write_ports << "\n"
              << "  max_dim_perline: { ";
    for (int d : layoutConfig.max_dim_perline)
      std::cout << d << " ";
    std::cout << "}\n  Factor order: { ";
    for (char f : layoutConfig.factor_order)
      std::cout << f << " ";
    std::cout << "}\n  Interline nest:\n";
    PrintNestLoopOrder(layoutConfig.interline, layoutConfig.factor_order);
    std::cout << "  Intraline nest:\n";
    PrintNestLoopOrder(layoutConfig.intraline, layoutConfig.factor_order);
    std::cout << "\n";
  }
  delete cc;
  return 0;
}
