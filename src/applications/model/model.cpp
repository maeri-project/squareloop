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

#include <fstream>

#include "util/accelergy_interface.hpp"
#include "util/banner.hpp"

#include "applications/model/model.hpp"
#include "layout/layout.hpp"
#include "crypto/crypto.hpp"

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

namespace application
{

template <class Archive>
void Model::serialize(Archive& ar, const unsigned int version)
{
  if (version == 0)
  {
    ar& BOOST_SERIALIZATION_NVP(workload_);
  }
}

Model::Model(config::CompoundConfig* config,
             std::string output_dir,
             std::string name) :
    name_(name)
{    
  auto rootNode = config->getRoot();

  // Version check
  if (rootNode.exists("architecture") && rootNode.lookup("architecture").exists("nodes"))
  {
    std::cerr << "ERROR: 'nodes' found as a sub-key in the architecture. The 'nodes' key is used by "
              << "the v0.4 timeloop front-end format, and will not be recognized by 'timeloop-...' "
              << "commands. Please use the timeloopfe front-end, which is documented at "
              << "https://github.com/Accelergy-Project/timeloop-accelergy-exercises and "
              << "https://timeloop.csail.mit.edu." << std::endl;
    exit(1);
  }

  // Model application configuration.
  auto_bypass_on_failure_ = false;
  std::string semi_qualified_prefix = name;

  if (rootNode.exists("model"))
  {
    auto model = rootNode.lookup("model");
    model.lookupValue("verbose", verbose_);
    model.lookupValue("auto_bypass_on_failure", auto_bypass_on_failure_);
    model.lookupValue("out_prefix", semi_qualified_prefix);
  }

  out_prefix_ = output_dir + "/" + semi_qualified_prefix;

  if (verbose_)
  {
    for (auto& line: banner)
      std::cout << line << std::endl;
    std::cout << std::endl;
  }

  // Problem configuration.
  auto problem = rootNode.lookup("problem");
  problem::ParseWorkload(problem, workload_);
  if (verbose_)
    std::cout << "Problem configuration complete." << std::endl;

  // std::cout << "CoefficientIDToName " << std::endl;
  // for (auto &key_pair: workload_.GetShape()->CoefficientIDToName)
  // {
  //   std::cout << key_pair.first << ": " << key_pair.second  << "=" <<  workload_.GetCoefficient(key_pair.first) << std::endl;
  // }
  // std::cout << std::endl;

  // std::cout << "FlattenedDimensionNameToID " << std::endl;
  // for (auto & key_pair: workload_.GetShape()->FlattenedDimensionNameToID){
  //   std::cout << key_pair.first << " " << key_pair.second << " ";//.first << " " << key_pair.second.second << " ";
  // };
  // std::cout << std::endl;

  // for(auto Name_RankName_Pair: workload_.GetShape()->DataSpaceNameToRankName){
  //   std::cout << Name_RankName_Pair.first << " ";
  //   for(auto in_vec: Name_RankName_Pair.second)
  //     std::cout << in_vec << " ";
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  // std::cout << " RankNameToFactorizedDimensionID " << std::endl;
  // for(auto Name_RankName_Pair: workload_.GetShape()->RankNameToFactorizedDimensionID){
  //   std::cout << Name_RankName_Pair.first << " ";
  //   for(auto in_vec: Name_RankName_Pair.second)
  //     std::cout << in_vec << " ";
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  // std::cout << "GetCoefficientID " << std::endl;
  // for (unsigned i = 0; i < workload_.GetShape()->NumFactorizedDimensions; ++i) { 
  //   std::cout << i << " ";//.first << " " << key_pair.second.second << " ";
  // };
  // std::cout << std::endl;

  // std::cout << "GetFactorizedBound " << std::endl;
  // for (unsigned i = 0; i < workload_.GetShape()->NumFactorizedDimensions; ++i) { 
  //   std::cout << workload_.GetFactorizedBound(i) << " ";//.first << " " << key_pair.second.second << " ";
  // };
  // std::cout << std::endl;

  // std::cout << "RankNameToCoefficient " << std::endl;
  // for (auto & key_pair: workload_.GetShape()->RankNameToCoefficient){
  //   std::cout << key_pair.first << ": (";//.first << " " << key_pair.second.second << " ";
  //   for(auto in_vec: key_pair.second)
  //     std::cout << in_vec << ", ";
  //   std::cout << ")" << std::endl;
  // };
  // std::cout << std::endl;

  // std::cout << "RankNameToDimensionName " << std::endl;
  // for (auto & key_pair: workload_.GetShape()->RankNameToDimensionName){
  //   std::cout << key_pair.first << ": (";//.first << " " << key_pair.second.second << " ";
  //   for(auto in_vec: key_pair.second)
  //     std::cout << in_vec << ", ";
  //   std::cout << ")" <<  std::endl;
  // };
  // std::cout << std::endl;



  // Architecture configuration.
  config::CompoundConfigNode arch;
  if (rootNode.exists("arch"))
  {
    arch = rootNode.lookup("arch");
  }
  else if (rootNode.exists("architecture"))
  {
    arch = rootNode.lookup("architecture");
  }
  
  bool is_sparse_topology = rootNode.exists("sparse_optimizations");
  arch_specs_ = model::Engine::ParseSpecs(arch, is_sparse_topology);

  if (rootNode.exists("ERT"))
  {
    auto ert = rootNode.lookup("ERT");
    if (verbose_)
      std::cout << "Found Accelergy ERT (energy reference table), replacing internal energy model." << std::endl;
    arch_specs_.topology.ParseAccelergyERT(ert);
    if (rootNode.exists("ART")){ // Nellie: well, if the users have the version of Accelergy that generates ART
      auto art = rootNode.lookup("ART");
      if (verbose_)
        std::cout << "Found Accelergy ART (area reference table), replacing internal area model." << std::endl;
      arch_specs_.topology.ParseAccelergyART(art);  
    }
  }
  else
  {
#ifdef USE_ACCELERGY
    // Call accelergy ERT with all input files
    if (arch.exists("subtree") || arch.exists("local"))
    {
      accelergy::invokeAccelergy(config->inFiles, semi_qualified_prefix, output_dir);
      std::string ertPath = out_prefix_ + ".ERT.yaml";
      auto ertConfig = new config::CompoundConfig(ertPath.c_str());
      auto ert = ertConfig->getRoot().lookup("ERT");
      if (verbose_)
        std::cout << "Generate Accelergy ERT (energy reference table) to replace internal energy model." << std::endl;
      arch_specs_.topology.ParseAccelergyERT(ert);
        
      std::string artPath = out_prefix_ + ".ART.yaml";
      auto artConfig = new config::CompoundConfig(artPath.c_str());
      auto art = artConfig->getRoot().lookup("ART");
      if (verbose_)
        std::cout << "Generate Accelergy ART (area reference table) to replace internal area model." << std::endl;
      arch_specs_.topology.ParseAccelergyART(art);
    }
#endif
  }

  // Sparse optimizations
  config::CompoundConfigNode sparse_optimizations;
  if (is_sparse_topology)
    sparse_optimizations = rootNode.lookup("sparse_optimizations");
      sparse_optimizations_ = new sparse::SparseOptimizationInfo(sparse::ParseAndConstruct(sparse_optimizations, arch_specs_));
  // characterize workload on whether it has metadata
  workload_.SetDefaultDenseTensorFlag(sparse_optimizations_->compression_info.all_ranks_default_dense);
  
  if (verbose_)
    std::cout << "Sparse optimization configuration complete." << std::endl;

  arch_props_ = new ArchProperties(arch_specs_);
  // Architecture constraints.
  config::CompoundConfigNode arch_constraints;

  if (arch.exists("constraints"))
    arch_constraints = arch.lookup("constraints");
  else if (rootNode.exists("arch_constraints"))
    arch_constraints = rootNode.lookup("arch_constraints");
  else if (rootNode.exists("architecture_constraints"))
    arch_constraints = rootNode.lookup("architecture_constraints");

  constraints_ = new mapping::Constraints(*arch_props_, workload_);
  constraints_->Parse(arch_constraints);

  if (verbose_)
    std::cout << "Architecture configuration complete." << std::endl;

  // Mapping configuration: expressed as a mapspace or mapping.
  auto mapping = rootNode.lookup("mapping");
  mapping_ = new Mapping(mapping::ParseAndConstruct(mapping, arch_specs_, workload_));
  if (verbose_)
    std::cout << "Mapping construction complete." << std::endl;

  // Validate mapping against the architecture constraints.
  if (!constraints_->SatisfiedBy(mapping_))
  {
    std::cerr << "ERROR: mapping violates architecture constraints." << std::endl;
    exit(1);
  }

  // crypto modeling
  std::cout << "Start Parsering Crypto" << std::endl;
  config::CompoundConfigNode compound_config_node_crypto;
  bool existing_crypto = rootNode.lookup("crypto", compound_config_node_crypto);
  crypto_ = new crypto::CryptoConfig();

  if (existing_crypto){
    crypto_ = crypto::ParseAndConstruct(compound_config_node_crypto);
    
    crypto_->crypto_initialized_ = true;
  }
  else{
    crypto_->crypto_initialized_ = false;
    std::cout << "No Crypto specified" << std::endl;
  }

  // layout modeling
  std::cout << "Start Parsering Layout" << std::endl;
  config::CompoundConfigNode compound_config_node_layout;
  bool existing_layout = rootNode.lookup("layout", compound_config_node_layout);
  
  if (existing_layout){
    std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>> externalPortMapping;
    for (auto i: arch_specs_.topology.StorageLevelNames())
        externalPortMapping.push_back({i, {arch_specs_.topology.GetStorageLevel(i)->num_ports.Get(), arch_specs_.topology.GetStorageLevel(i)->num_ports.Get()}});

    layout_ = layout::ParseAndConstruct(compound_config_node_layout, workload_, externalPortMapping);
    
    layout_initialized_ = true;

    layout::PrintOverallLayout(layout_);
  }
  else{
    layout_initialized_ = false;
    std::cout << "No Layout specified, so using bandwidth based modeling" << std::endl;
  }
  
}

Model::~Model()
{
  if (mapping_)
    delete mapping_;

  if (arch_props_)
    delete arch_props_;
  
  if (constraints_)
    delete constraints_;

  if (sparse_optimizations_)
    delete sparse_optimizations_;
}

// Run the evaluation.
Model::Stats Model::Run()
{
  model::Engine engine;
  engine.Spec(arch_specs_);

  auto level_names = arch_specs_.topology.LevelNames();

  auto& mapping = *mapping_;
    
  // Optional feature: if the given mapping does not fit in the available
  // hardware resources, automatically bypass storage level(s) to make it
  // fit. This avoids mapping failures and instead substitutes the given
  // mapping with one that fits but is higher cost and likely sub-optimal.
  // *However*, this only covers capacity failures due to temporal factors,
  // not instance failures due to spatial factors. It also possibly
  // over-corrects since it bypasses *all* data_spaces at a failing level,
  // while it's possible that bypassing a subset of data_spaces may have
  // caused the mapping to fit.
  if (auto_bypass_on_failure_)
  {
    auto pre_eval_status = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, false);
    for (unsigned level = 0; level < pre_eval_status.size(); level++)
      if (!pre_eval_status[level].success)
      {
        if (verbose_)
          std::cerr << "WARNING: couldn't map level " << level_names.at(level) << ": "
                    << pre_eval_status[level].fail_reason << ", auto-bypassing."
                    << std::endl;
        for (unsigned pvi = 0; pvi < workload_.GetShape()->NumDataSpaces; pvi++)
          // Ugh... mask is offset-by-1 because level 0 is the arithmetic level.
          mapping.datatype_bypass_nest.at(pvi).reset(level-1);
      }
  }
  
  if (layout_initialized_){ 
    auto eval_status = engine.Evaluate(mapping, workload_, layout_, sparse_optimizations_, crypto_);
    for (unsigned level = 0; level < eval_status.size(); level++)
    {
      if (!eval_status[level].success)
      {
        std::cerr << "ERROR: couldn't map level " << level_names.at(level) << ": "
                  << eval_status[level].fail_reason << std::endl;
        exit(1);
      }
    }
  }else{
    auto eval_status = engine.Evaluate(mapping, workload_, sparse_optimizations_, crypto_);    
    for (unsigned level = 0; level < eval_status.size(); level++)
    {
      if (!eval_status[level].success)
      {
        std::cerr << "ERROR: couldn't map level " << level_names.at(level) << ": "
                  << eval_status[level].fail_reason << std::endl;
        exit(1);
      }
    }
  }

  
  // if (!std::accumulate(success.begin(), success.end(), true, std::logical_and<>{}))
  // {
  //   std::cout << "Illegal mapping, evaluation failed." << std::endl;
  //   return;
  // }

  std::stringstream map_txt;
  std::stringstream stats_txt;
  if (engine.IsEvaluated())
  {
    if (!sparse_optimizations_->no_optimization_applied)
    {   
      std::cout << "Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << engine.Utilization()
              << " | pJ/Algorithmic-Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << engine.Energy() /
      engine.GetTopology().AlgorithmicComputes()
              << " | pJ/Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << engine.Energy() /
      engine.GetTopology().ActualComputes()
              << " | Cycles = " << engine.Cycles()  << std::endl;

    }
    else
    {
      std::cout << "Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << engine.Utilization()
                << " | pJ/Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << engine.Energy() /
      engine.GetTopology().ActualComputes()
                << " | Cycles = " << engine.Cycles()  << std::endl;
    }

    mapping.PrettyPrint(map_txt,
                        arch_specs_.topology.StorageLevelNames(),
                        engine.GetTopology().UtilizedCapacities(),
                        engine.GetTopology().TileSizes());

    stats_txt << engine << std::endl;
  }

  // Print the engine stats and mapping to an XML file
  std::stringstream xml_str;
  boost::archive::xml_oarchive ar(xml_str);
  ar << BOOST_SERIALIZATION_NVP(engine);
  ar << BOOST_SERIALIZATION_NVP(mapping);
  const Model* a = this;
  ar << BOOST_SERIALIZATION_NVP(a);

  // Print the mapping in Tenssella input format.
  std::stringstream tenssella_out;
  mapping.PrintTenssella(tenssella_out);

  Stats stats;
  stats.map_string = map_txt.str();
  stats.stats_string = stats_txt.str();
  stats.xml_map_and_stats_string = xml_str.str();
  stats.tensella_string = tenssella_out.str();

  stats.cycles = engine.Cycles();
  stats.energy = engine.Energy();
  return stats;
}

} // namespace application
