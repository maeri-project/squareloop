#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

// --- Dummy definitions for problem::Shape and problem::GetShape() ---
namespace problem {
  struct Shape {
    using FlattenedDimensionID = int;
    static const int NumFlattenedDimensions = 100; // dummy default value
  };

  const Shape* GetShape() {
    static Shape shape;
    return &shape;
  }
}

// --- Original Nest definition (DO NOT CHANGE) ---
// (Note: target and type fields have been removed.)
class Nest
{
public:
  // Skew specs.
  struct SkewDescriptor
  {
    struct Term
    {
      struct DimSpec
      {
        problem::Shape::FlattenedDimensionID dimension = problem::GetShape()->NumFlattenedDimensions;
        bool is_spatial = false;
      };
      // Each skew term can have a constant, a loop bound, and a loop variable.
      int constant = 1;
      DimSpec variable;
      DimSpec bound;
      char dim_char = '\0';  // record the factor letter (e.g. 'M', 'K', etc.)
    };
    std::vector<Term> terms;
    int modulo = 0;
  };

  // One or more skew descriptors.
  std::vector<SkewDescriptor> skew_descriptors;
};

// --- New structure: Layout ---
// This structure holds per-target information. Note that there is no "type" field,
// because the two separate nests (interline and intraline) make that redundant.
struct Layout
{
  std::string target;
  Nest interline;   // single interline nest
  Nest intraline;   // single intraline nest
  int num_read_ports = 1;   // from the interline specification (default 1)
  int num_write_ports = 1;  // from the interline specification (default 1)
  std::vector<int> max_dim_perline; // computed from the intraline nest
};

// --- Helper: Parse the order mapping string ---
// Expected format, for example: "C:0, M:1, R:2, S:3, N:4, P:5, Q:6"
std::unordered_map<char, int> parseOrderMapping(const std::string &mappingStr)
{
  std::unordered_map<char, int> mapping;
  std::istringstream iss(mappingStr);
  std::string token;
  while (std::getline(iss, token, ','))
  {
    // Remove any spaces.
    token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
    if (token.empty()) continue;
    size_t colonPos = token.find(':');
    if (colonPos != std::string::npos && colonPos > 0 && colonPos < token.size()-1)
    {
      char dim = token[0];
      int order = std::stoi(token.substr(colonPos + 1));
      mapping[dim] = order;
    }
  }
  return mapping;
}

// --- The parser ---
// This function reads a YAML file with entries such as:
/*
layout:
  - target: MainMemory
    type: interline
    factors: R=3 S=3 P=7 Q=7 C=3 M=1 N=1
    permutation: SR CQP MN
    num_read_ports: 1
    num_write_ports: 2

  - target: GlobalBuffer
    type: interline
    factors: R=1 S=1 P=7 Q=4 C=3 M=1 N=1
    permutation: SR CQP MN
    num_read_ports: 1
    num_write_ports: 2

  - target: GlobalBuffer
    type: intraline
    factors: R=1 S=1 P=1 Q=2 C=1 M=1 N=1
    permutation: SR CQP MN

  - target: RegisterFile
    type: interline
    factors: R=1 S=1 P=1 Q=1 C=3 M=1 N=1
    permutation: SR CQP MN
    num_read_ports: 1
    num_write_ports: 2
*/
std::unordered_map<std::string, Layout> parseLayoutYaml(const std::string &filename)
{
  std::unordered_map<std::string, Layout> layouts;
  YAML::Node config = YAML::LoadFile(filename);
  if (!config["layout"])
  {
    std::cerr << "No 'layout' node in YAML file.\n";
    return layouts;
  }

  // Process each layout block.
  for (const auto &layoutNode : config["layout"])
  {
    // Extract target and type from the YAML node.
    std::string target = layoutNode["target"].as<std::string>();
    std::string type = layoutNode["type"].as<std::string>(); // either "interline" or "intraline"

    // Create a Nest from this YAML entry.
    Nest nest;
    // --- Parse the 'factors' field ---
    // For example: "R=3 S=3 P=7 Q=7 C=3 M=1 N=1"
    std::unordered_map<char, int> factors;
    std::istringstream factorsStream(layoutNode["factors"].as<std::string>());
    std::string factorToken;
    while (factorsStream >> factorToken)
    {
      size_t eqPos = factorToken.find('=');
      if (eqPos != std::string::npos && eqPos > 0 && eqPos < factorToken.size()-1)
      {
        char factor = factorToken[0];
        int value = std::stoi(factorToken.substr(eqPos+1));
        factors[factor] = value;
      }
    }

    // --- Parse the 'permutation' field ---
    // For example: "SR CQP MN" where each space-separated group becomes one SkewDescriptor.
    std::istringstream permStream(layoutNode["permutation"].as<std::string>());
    std::string group;
    while (permStream >> group)
    {
      Nest::SkewDescriptor skew;
      for (char factor : group)
      {
        Nest::SkewDescriptor::Term term;
        term.dim_char = factor;
        if (factors.count(factor))
          term.constant = factors[factor];
        // A dummy mapping: map 'A'-'Z' to 0-25.
        if (factor >= 'A' && factor <= 'Z')
        {
          term.variable.dimension = factor - 'A';
          term.bound.dimension = factor - 'A';
          // Mark some factors as spatial if desired.
          if (factor == 'R' || factor == 'S' || factor == 'P' || factor == 'Q')
          {
            term.variable.is_spatial = true;
            term.bound.is_spatial = true;
          }
        }
        skew.terms.push_back(term);
      }
      nest.skew_descriptors.push_back(skew);
    }

    // --- Group the nest into a Layout based on its target ---
    if (layouts.find(target) == layouts.end())
    {
      Layout l;
      l.target = target;
      layouts[target] = l;
    }

    // For an interline nest, record port counts (if provided) and assign if not already set.
    if (type == "interline")
    {
      int nr = (layoutNode["num_read_ports"]) ? layoutNode["num_read_ports"].as<int>() : 1;
      int nw = (layoutNode["num_write_ports"]) ? layoutNode["num_write_ports"].as<int>() : 1;
      if (layouts[target].interline.skew_descriptors.empty())
      {
        layouts[target].interline = nest;
        layouts[target].num_read_ports = nr;
        layouts[target].num_write_ports = nw;
      }
    }
    // For an intraline nest, assign it if not already set.
    else if (type == "intraline")
    {
      if (layouts[target].intraline.skew_descriptors.empty())
      {
        layouts[target].intraline = nest;
      }
    }
    else
    {
      // Default to interline if type is unrecognized.
      if (layouts[target].interline.skew_descriptors.empty())
      {
        layouts[target].interline = nest;
      }
    }
  }
  return layouts;
}

// --- Function to print the nested loop order for a given Nest ---
// Output format:
//   iter->dimension=<order>-<dim> in [0, <bound>, 1) iter->residual_end=<bound>
void printNestLoopOrder(const Nest &nest, const std::unordered_map<char, int> &orderMap)
{
  for (const auto &skew : nest.skew_descriptors)
  {
    for (const auto &term : skew.terms)
    {
      int order = -1;
      if (orderMap.count(term.dim_char))
        order = orderMap.at(term.dim_char);
      std::cout << "iter->dimension=" << order << "-" << term.dim_char
                << " in [0, " << term.constant << ", 1) iter->residual_end="
                << term.constant << "\n";
    }
  }
}

//
// --- Main ---
//
// Usage example:
//   ./nest_parser layout.yaml "C:0, M:1, R:2, S:3, N:4, P:5, Q:6"
//
// If no order mapping string is provided, a default order is used.
int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <layout.yaml> [order_mapping_string]\n";
    return 1;
  }

  std::string filename = argv[1];
  std::string orderMappingStr = "C:0, M:1, R:2, S:3, N:4, P:5, Q:6";
  if (argc > 2)
    orderMappingStr = argv[2];
  std::unordered_map<char, int> specifiedOrder = parseOrderMapping(orderMappingStr);

  // Parse the YAML layout.
  auto layouts = parseLayoutYaml(filename);

  // --- For any target missing an intraline or interline nest, create a default nest ---
  // We use the sorted order mapping (by order value) to define the loop order.
  std::vector<std::pair<int, char>> sortedFactors;
  for (const auto &p : specifiedOrder)
    sortedFactors.push_back({p.second, p.first});
  std::sort(sortedFactors.begin(), sortedFactors.end(),
            [](const std::pair<int, char> &a, const std::pair<int, char> &b) {
              return a.first < b.first;
            });

  for (auto &entry : layouts)
  {
    Layout &layout = entry.second;
    // If the intraline nest is empty, create a default one.
    if (layout.intraline.skew_descriptors.empty())
    {
      Nest defaultIntraline;
      Nest::SkewDescriptor skew;
      for (const auto &pf : sortedFactors)
      {
        char factor = pf.second;
        Nest::SkewDescriptor::Term term;
        term.dim_char = factor;
        term.constant = 1;
        if (factor >= 'A' && factor <= 'Z')
        {
          term.variable.dimension = factor - 'A';
          term.bound.dimension = factor - 'A';
        }
        skew.terms.push_back(term);
      }
      defaultIntraline.skew_descriptors.push_back(skew);
      layout.intraline = defaultIntraline;
    }
    // If the interline nest is empty, create a default one.
    if (layout.interline.skew_descriptors.empty())
    {
      Nest defaultInterline;
      Nest::SkewDescriptor skew;
      for (const auto &pf : sortedFactors)
      {
        char factor = pf.second;
        Nest::SkewDescriptor::Term term;
        term.dim_char = factor;
        term.constant = 1;
        if (factor >= 'A' && factor <= 'Z')
        {
          term.variable.dimension = factor - 'A';
          term.bound.dimension = factor - 'A';
        }
        skew.terms.push_back(term);
      }
      defaultInterline.skew_descriptors.push_back(skew);
      layout.interline = defaultInterline;
    }
  }

  // --- Compute max_dim_perline for each target ---
  // For each target, compute max_dim_perline from its intraline nest.
  // (We assume the first (and only) intraline nest defines the per-line factors.)
  for (auto &entry : layouts)
  {
    Layout &layout = entry.second;
    std::vector<int> maxDim(sortedFactors.size(), 1);
    if (!layout.intraline.skew_descriptors.empty())
    {
      const Nest &intra = layout.intraline;
      for (const auto &skew : intra.skew_descriptors)
      {
        for (const auto &term : skew.terms)
        {
          if (specifiedOrder.count(term.dim_char))
          {
            int order = specifiedOrder.at(term.dim_char);
            for (size_t i = 0; i < sortedFactors.size(); ++i)
            {
              if (sortedFactors[i].first == order)
              {
                maxDim[i] = term.constant;
                break;
              }
            }
          }
        }
      }
    }
    layout.max_dim_perline = maxDim;
  }

  // --- Display the results ---
  for (const auto &entry : layouts)
  {
    const Layout &layout = entry.second;
    std::cout << "Target: " << layout.target << "\n";
    std::cout << "  num_read_ports: " << layout.num_read_ports
              << ", num_write_ports: " << layout.num_write_ports << "\n";
    std::cout << "  max_dim_perline: { ";
    for (size_t i = 0; i < layout.max_dim_perline.size(); ++i)
      std::cout << layout.max_dim_perline[i] << " ";
    std::cout << "}\n";

    std::cout << "  Interline nest:\n";
    printNestLoopOrder(layout.interline, specifiedOrder);
    std::cout << "  Intraline nest:\n";
    printNestLoopOrder(layout.intraline, specifiedOrder);
  }

  return 0;
}
