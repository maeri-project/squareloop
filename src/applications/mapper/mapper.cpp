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
#include <thread>
#include <mutex>
#include <iomanip>
#include <ncurses.h>

#include "util/accelergy_interface.hpp"

#include "applications/mapper/mapper.hpp"
#include "layout/layout.hpp"
#include "layoutspaces/layoutspace.hpp"
#include "crypto/crypto.hpp"

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

namespace application
{

template <class Archive>
void Mapper::serialize(Archive& ar, const unsigned int version)
{
  if (version == 0)
  {
    ar& BOOST_SERIALIZATION_NVP(workload_);
  }
}

Mapper::Mapper(config::CompoundConfig* config,
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

  // Problem configuration.
  auto problem = rootNode.lookup("problem");
  problem::ParseWorkload(problem, workload_);
  std::cout << "Problem configuration complete." << std::endl;
  std::cout << "Print out overall CoefficientIDToName of the given problem" << std::endl;
  for (auto pro: workload_.GetShape()->CoefficientIDToName){
    std::cout << pro.first << " " << pro.second  << " " ;
  }
  std::cout << std::endl;
  for (auto pro: workload_.GetShape()->DefaultCoefficients){
    std::cout << pro.first << " " << pro.second  << " " ;
  }
  std::cout << std::endl;

  std::cout << "Print out overall FactorizedDimensionNameToID of the given problem" << std::endl;
  for (auto pro: workload_.GetShape()->FactorizedDimensionNameToID){
    std::cout << pro.first << ":" << pro.second << " ";
  }
  std::cout << std::endl;
  std::cout << "Print out overall workload_.GetFactorizedBounds()->GetCoordinates from problem" << std::endl;
  for (int dim = 0; dim < int(workload_.GetShape()->NumFlattenedDimensions); dim++)
  {
    std::cout << workload_.GetShape()->FlattenedDimensionIDToName.at(dim) << " ";
  }
  std::cout << std::endl;
  for (auto pro: workload_.GetFactorizedBounds().GetCoordinates()){
    std::cout << pro << " ";
  }
  std::cout << std::endl;

  std::cout << "Print out DataSpaceIDToDimensionIDVector from problem" << std::endl;
  for (unsigned index =0; index < workload_.GetShape()->NumDataSpaces; index++){
    std::cout << workload_.GetShape()->DataSpaceIDToName.at(index) << ":";
    for(auto pro2: workload_.GetShape()->DataSpaceIDToDimensionIDVector[index]){
      std::cout << pro2  << " ";
    }
    std::cout << std::endl;
    index++;
  }
  std::cout << std::endl;
  std::cout << "Overall Workload Dimension" << std::endl;
  for (int dim = 0; dim < int(workload_.GetShape()->NumFlattenedDimensions); dim++)
  {
    std::cout << workload_.GetShape()->FlattenedDimensionIDToName.at(dim) << " = "
              << workload_.GetFlattenedBound(dim) << std::endl;
  }
  std::cout << std::endl;
  std::cout << "Print out overall DataSpaceIDToName of the given problem" << std::endl;
  for (auto pro:  workload_.GetShape()->DataSpaceIDToName){
    std::cout << pro.first << " " << pro.second << " ";
  }
  std::cout << std::endl;

  // std::cout << "Print out overall GetCoIteratedDimensions of the given problem" << std::endl;
  // for (const auto& pro: shape_problem->GetCoIteratedDimensions()){
  //   std::cout << pro << " ";
  // }

  // Mapper (this application) configuration.
  auto mapper = rootNode.lookup("mapper");
  std::string semi_qualified_prefix = name;
  mapper.lookupValue("out_prefix", semi_qualified_prefix);
  out_prefix_ = output_dir + "/" + semi_qualified_prefix;

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
    std::cout << "Found Accelergy ERT (energy reference table), replacing internal energy model." << std::endl;
    arch_specs_.topology.ParseAccelergyERT(ert);
    if (rootNode.exists("ART")){ // Nellie: well, if the users have the version of Accelergy that generates ART
      auto art = rootNode.lookup("ART");
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
      std::cout << "Generate Accelergy ERT (energy reference table) to replace internal energy model." << std::endl;
      arch_specs_.topology.ParseAccelergyERT(ert);

      std::string artPath = out_prefix_ + ".ART.yaml";
      auto artConfig = new config::CompoundConfig(artPath.c_str());
      auto art = artConfig->getRoot().lookup("ART");
      std::cout << "Generate Accelergy ART (area reference table) to replace internal area model." << std::endl;
      arch_specs_.topology.ParseAccelergyART(art);
    }
#endif
  }

  std::cout << "Architecture configuration complete." << std::endl;

  // Sparse optimizations
  config::CompoundConfigNode sparse_optimizations;
  if (is_sparse_topology)
    sparse_optimizations = rootNode.lookup("sparse_optimizations");

  sparse_optimizations_ = new sparse::SparseOptimizationInfo(sparse::ParseAndConstruct(sparse_optimizations, arch_specs_));
  // characterize workload on whether it has metadata
  workload_.SetDefaultDenseTensorFlag(sparse_optimizations_->compression_info.all_ranks_default_dense);

  std::cout << "Sparse optimization configuration complete." << std::endl;

  // Mapper (this application) configuration. (the rest)

  num_threads_ = std::thread::hardware_concurrency();
  if (mapper.lookupValue("num_threads", num_threads_))
  {
    std::cout << "Using threads = " << num_threads_ << std::endl;
  }
  else
  {
    std::cout << "Using all available hardware threads = " << num_threads_ << std::endl;
  }

  std::string metric;
  std::vector<std::string> raw_metrics;
  if (mapper.lookupValue("optimization_metric", metric))
  {
    raw_metrics = { metric };
  }
  else if (mapper.exists("optimization_metrics"))
  {
    mapper.lookupArrayValue("optimization_metrics", raw_metrics);
  }
  else
  {
    std::cerr << "WARNING: no optimization metric(s) specified, using edp as default." << std::endl;
    raw_metrics = { "edp" };
  }

  for (auto& metric: raw_metrics)
  {
    // Special-case: if any metric is "ordered_accesses" expand it into a list
    // of "access_X" strings.
    if (metric == "ordered_accesses")
    {
      auto num_levels = arch_specs_.topology.NumStorageLevels();
      for (unsigned i = num_levels-1; i < num_levels; i--)
      {
        optimization_metrics_.push_back(std::string("accesses-") + std::to_string(i));
      }
    }
    else
    {
      optimization_metrics_.push_back(metric);
    }
  }

  // Search size (divide between threads).
  std::uint32_t search_size = 0;
  mapper.lookupValue("search_size", search_size);
  mapper.lookupValue("search_size", search_size); // backwards compatibility.
  if (search_size > 0)
    search_size = 1 + (search_size - 1) / num_threads_;
  search_size_ = static_cast<uint128_t>(search_size);

  // Number of consecutive invalid mappings to trigger termination.
  timeout_ = 1000;
  mapper.lookupValue("timeout", timeout_);
  mapper.lookupValue("heartbeat", timeout_); // backwards compatibility.

  // Number of suboptimal valid mappings to trigger victory
  // (do NOT divide between threads).
  victory_condition_ = 500;
  mapper.lookupValue("victory_condition", victory_condition_);

  // Inter-thread sync interval.
  std::uint32_t sync_interval = 0;
  mapper.lookupValue("sync_interval", sync_interval);
  sync_interval_ = static_cast<uint128_t>(sync_interval);

  // Inter-thread sync interval.
  std::uint32_t log_interval = 1;
  mapper.lookupValue("log_interval", log_interval);
  log_interval_ = static_cast<uint128_t>(log_interval);

  int32_t max_temporal_loops_in_a_mapping = -1;
  mapper.lookupValue("max_temporal_loops_in_a_mapping", max_temporal_loops_in_a_mapping);
  max_temporal_loops_in_a_mapping_ = static_cast<int32_t>(max_temporal_loops_in_a_mapping);

  // Misc.
  log_orojenesis_mappings_ = false;
  mapper.lookupValue("log_orojenesis_mappings", log_orojenesis_mappings_);

  log_mappings_yaml_ = false;
  mapper.lookupValue("log_mappings_yaml", log_mappings_yaml_);

  log_mappings_verbose_ = false;
  mapper.lookupValue("log_mappings_verbose", log_mappings_verbose_);

  // Misc.
  log_all_mappings_ = false;
  mapper.lookupValue("log_all_mappings", log_all_mappings_);

  log_stats_ = false;
  mapper.lookupValue("log_stats", log_stats_);

  log_suboptimal_ = false;
  mapper.lookupValue("log_suboptimal", log_suboptimal_);
  mapper.lookupValue("log_all", log_suboptimal_); // backwards compatibility.

  live_status_ = false;
  mapper.lookupValue("live_status", live_status_);

  diagnostics_on_ = false;
  mapper.lookupValue("diagnostics", diagnostics_on_);

  penalize_consecutive_bypass_fails_ = false;
  mapper.lookupValue("penalize_consecutive_bypass_fails", penalize_consecutive_bypass_fails_);

  emit_whoop_nest_ = false;
  mapper.lookupValue("emit_whoop_nest", emit_whoop_nest_);

  std::cout << "Mapper configuration complete." << std::endl;

  // MapSpace configuration.
  config::CompoundConfigNode arch_constraints;
  config::CompoundConfigNode mapspace;

  // Architecture constraints.
  if (arch.exists("constraints"))
    arch_constraints = arch.lookup("constraints");
  else if (rootNode.exists("arch_constraints"))
    arch_constraints = rootNode.lookup("arch_constraints");
  else if (rootNode.exists("architecture_constraints"))
    arch_constraints = rootNode.lookup("architecture_constraints");

  // Mapspace constraints.
  if (rootNode.exists("mapspace") && rootNode.exists("mapspace_constraints"))
  {
    std::cerr << "ERROR: found both \"mapspace\" and \"mapspace_constraints\" "
              << "directive. Please use either for specifying "
              << "mapspace constraints." << std::endl;
    exit(1);
  }

  if (rootNode.exists("mapspace"))
    mapspace = rootNode.lookup("mapspace");
  else if (rootNode.exists("mapspace_constraints"))
    mapspace = rootNode.lookup("mapspace_constraints");
  // else
  // {
  //   std::cerr << "ERROR: found neither \"mapspace\" nor \"mapspace_constraints\" "
  //             << "directive. To run the mapper without any constraints set "
  //             << "mapspace_constraints as an empty list []." << std::endl;
  //   exit(1);
  // }

  bool filter_spatial_fanout = sparse_optimizations_->action_spatial_skipping_info.size() == 0;
  mapspace_ = mapspace::ParseAndConstruct(mapspace, arch_constraints, arch_specs_, workload_, filter_spatial_fanout);
  split_mapspaces_ = mapspace_->Split(num_threads_);

  std::cout << "Mapspace construction complete." << std::endl;

  // Search configuration.
  auto search = rootNode.lookup("mapper");
  for (unsigned t = 0; t < num_threads_; t++)
  {
    search_.push_back(search::ParseAndConstruct(search, split_mapspaces_.at(t), t));
  }
  std::cout << "Search configuration complete." << std::endl;
  // Store the complete configuration in a string.
  if (config->hasLConfig())
  {
    std::size_t len;
    FILE* cfg_stream = open_memstream(&cfg_string_, &len);
    auto& lconfig = config->getLConfig();
    lconfig.write(cfg_stream);
    fclose(cfg_stream);
  }
  else
  {
    cfg_string_ = nullptr;
  }

  // crypto modeling
  std::cout << "Start Parsering Crypto" << std::endl;
  config::CompoundConfigNode compound_config_node_crypto;
  bool existing_crypto = rootNode.lookup("crypto", compound_config_node_crypto);
  crypto_ = new crypto::CryptoConfig();

  if (existing_crypto){
    crypto_ = crypto::ParseAndConstruct(compound_config_node_crypto);

    crypto_->crypto_initialized_ = true;

    std::cout << "Crypto Configuration:\n";
    std::cout << "  Name: " << crypto_->name << "\n";
    std::cout << "  Family: " << crypto_->family << "\n";
    std::cout << "  Datapath: " << crypto_->datapath << "\n";
    std::cout << "  Auth Additional Cycle Per Block: " << crypto_->auth_additional_cycle_per_block << "\n";
    std::cout << "  Auth Additional Energy Per Block: " << crypto_->auth_additional_energy_per_block << "\n";
    std::cout << "  Auth Cycle Per Datapath: " << crypto_->auth_cycle_per_datapath << "\n";
    std::cout << "  Auth Enc Parallel: " << (crypto_->auth_enc_parallel ? "true" : "false") << "\n";
    std::cout << "  Auth Energy Per Datapath: " << crypto_->auth_energy_per_datapath << "\n";
    std::cout << "  Enc Cycle Per Datapath: " << crypto_->enc_cycle_per_datapath << "\n";
    std::cout << "  Enc Energy Per Datapath: " << crypto_->enc_energy_per_datapath << "\n";
    std::cout << "  Hash Size: " << crypto_->hash_size << "\n";
    std::cout << "  Xor Cycle: " << crypto_->xor_cycle << "\n";
    std::cout << "  Xor Energy Per Datapath: " << crypto_->xor_energy_per_datapath << "\n";
  }
  else{
    std::cout << "No Crypto specified" << std::endl;
  }

  // layout modeling
  std::cout << "Start Parsering Layout" << std::endl;
  config::CompoundConfigNode compound_config_node_layout;
  bool existing_layout = rootNode.lookup("layout", compound_config_node_layout);

  std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>> externalPortMapping;
  for (auto i: arch_specs_.topology.StorageLevelNames()){
      externalPortMapping.push_back({i, {arch_specs_.topology.GetStorageLevel(i)->num_ports.Get(), arch_specs_.topology.GetStorageLevel(i)->num_ports.Get()}});
    std::cout << "Storage Level " << i << " has " << arch_specs_.topology.GetStorageLevel(i)->num_ports.Get() << " ports" << std::endl;
  }

  if (existing_layout){
    layout_ = layout::ParseAndConstruct(compound_config_node_layout, workload_, externalPortMapping);

    layout_initialized_ = true;
    layout::PrintOverallLayout(layout_);
  }
  else{
    layout_initialized_ = false;
    std::cout << "No Layout specified, using concordant layout with authblock_lines searching." << std::endl;
    layout_ = layout::InitializeDummyLayout(workload_, externalPortMapping);
    layout::PrintOverallLayout(layout_);
  }
}

Mapper::~Mapper()
{
  if (mapspace_)
  {
    delete mapspace_;
  }

  if (sparse_optimizations_)
  {
    delete sparse_optimizations_;
  }

  for (auto& search: search_)
  {
    if (search)
    {
      delete search;
    }
  }
}

EvaluationResult Mapper::GetGlobalBest()
{
  return global_best_;
}

// ---------------
// Run the mapper.
// ---------------
Mapper::Result Mapper::Run()
{
  // Output file names.
  std::string log_file_name = out_prefix_ + ".log";
  std::string map_cfg_file_name = out_prefix_ + ".map.cfg";
  std::string orojenesis_prefix = out_prefix_ + ".orojenesis";

  // Prepare live status/log stream.
  std::ofstream log_file;
  std::stringstream orojenesis_stream;

  // std::streambuf* streambuf_cout = std::cout.rdbuf();
  std::streambuf* streambuf_cerr = std::cerr.rdbuf();

  if (live_status_)
  {
    log_file.open(log_file_name);
    // std::cout.rdbuf(log_file.rdbuf());
    std::cerr.rdbuf(log_file.rdbuf());

    initscr();
    cbreak();
    noecho();
    clear();

    std::stringstream line0, line1, line2, line3, line4, line5;
    line0 << "================================================================================";
    line1 << "                                TIMELOOP MAPPER";
    line2 << "================================================================================";
    line3 << std::setw(3) << "TID" << std::setw(11) << "Total" << std::setw(11) << "Invalid"
          << std::setw(11) << "Valid" <<  std::setw(11) << "Consec." << std::setw(11) << "Last"
          << std::setw(11) << "Opt.util" << std::setw(11) << "Opt.energy";
    line4 << std::setw(3) << " " << std::setw(11) << " " << std::setw(11) << " "
          << std::setw(11) << " " <<  std::setw(11) << "invalid" << std::setw(11) << "update";
    line5 << "--------------------------------------------------------------------------------";
    mvaddstr(0, 0, line0.str().c_str());
    mvaddstr(1, 0, line1.str().c_str());
    mvaddstr(2, 0, line2.str().c_str());
    mvaddstr(3, 0, line3.str().c_str());
    mvaddstr(4, 0, line4.str().c_str());
    mvaddstr(5, 0, line5.str().c_str());
    refresh();
  }

  // Prepare the threads.
  std::mutex mutex;
  std::vector<MapperThread*> threads_;
  for (unsigned t = 0; t < num_threads_; t++)
  {
    threads_.push_back(new MapperThread(t, search_.at(t),
                                        split_mapspaces_.at(t),
                                        &mutex,
                                        search_size_,
                                        timeout_,
                                        victory_condition_,
                                        max_temporal_loops_in_a_mapping_,
                                        sync_interval_,
                                        log_interval_,
                                        log_orojenesis_mappings_,
                                        log_mappings_yaml_,
                                        log_mappings_verbose_,
                                        log_all_mappings_,
                                        log_stats_,
                                        log_suboptimal_,
                                        live_status_ ? log_file : std::cerr,
                                        orojenesis_stream,
                                        orojenesis_prefix,
                                        live_status_,
                                        diagnostics_on_,
                                        penalize_consecutive_bypass_fails_,
                                        optimization_metrics_,
                                        arch_specs_,
                                        workload_,
                                        layout_,
                                        layout_initialized_,
                                        sparse_optimizations_,
                                        crypto_,
                                        &best_));
  }

  // Launch the threads.
  for (unsigned t = 0; t < num_threads_; t++)
  {
    threads_.at(t)->Start();
  }

  // Wait for the threads to join.
  for (unsigned t = 0; t < num_threads_; t++)
  {
    threads_.at(t)->Join();
  }

  // Close log and end curses.
  if (live_status_)
  {
    // std::cout.rdbuf(streambuf_cout);
    std::cerr.rdbuf(streambuf_cerr);
    log_file.close();

    mvaddstr(LINES-1, 0, "Press any key to exit.");
    getch();
    endwin();
  }

  // Diagnostics.
  if (diagnostics_on_)
  {
    // Aggregate diagnostic data from all threads.
    std::map<FailClass, std::map<unsigned, FailInfo>> fail_stats;

    for (unsigned t = 0; t < num_threads_; t++)
    {
      for (auto& i: threads_.at(t)->GetStats().fail_stats)
      {
        auto& thread_fail_class = i.first;
        auto& thread_fail_bucket = i.second;

        auto fail_bucket_it = fail_stats.find(thread_fail_class);
        if (fail_bucket_it == fail_stats.end())
        {
          // We've never seen this fail class before.
          fail_stats[thread_fail_class] = thread_fail_bucket;
        }
        else
        {
          auto& fail_bucket = fail_bucket_it->second;

          // We've seen this fail class. Walk through each level in this fail bucket.
          for (auto& j: thread_fail_bucket)
          {
            auto& thread_fail_level_id = j.first;
            auto& thread_fail_info = j.second;

            auto fail_info_it = fail_bucket.find(thread_fail_level_id);
            if (fail_info_it == fail_bucket.end())
            {
              // We haven't seen this level within this fail bucket.
              fail_bucket[thread_fail_level_id] = thread_fail_info;
            }
            else
            {
              // We've seen this level within this fail bucket.
              fail_info_it->second.count += thread_fail_info.count;
            }
          }
        }
      }
    }


    // Print.
    std::cout << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "               BEGIN DIAGNOSTICS               " << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    for (auto& i: fail_stats)
    {
      auto& fail_class = i.first;
      auto& fail_bucket = i.second;

      std::cout << "Fail class: " << fail_class << std::endl;
      for (auto& j: fail_bucket)
      {
        std::cout << std::endl;
        std::cout << "  Level: " << arch_specs_.topology.GetLevel(j.first)->level_name << std::endl;
        std::cout << "    Fail count: " << j.second.count << std::endl;
        std::cout << "    Sample mapping that experienced this fail class:" << std::endl;

        auto& mapping = j.second.mapping;

        model::Engine engine;
        engine.Spec(arch_specs_);

        if (layout_initialized_){ // ToDo: @Jianming modify here
          engine.Evaluate(mapping, workload_, layout_, sparse_optimizations_, crypto_, false);
        }else
          engine.Evaluate(mapping, workload_, sparse_optimizations_, crypto_, false);

        mapping.PrettyPrint(std::cout, arch_specs_.topology.StorageLevelNames(),
                            engine.GetTopology().GetStats().utilized_capacities,
                            engine.GetTopology().GetStats().tile_sizes, "      ");

        std::cout << "    Fail reason: " << j.second.reason << std::endl;
        std::cout << std::endl;
      }
    }

    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "                 END DIAGNOSTICS               " << std::endl;
    std::cout << "===============================================" << std::endl;
  }

  // Select the best mapping from each thread.
  for (unsigned t = 0; t < num_threads_; t++)
  {
    // Each thread tracks its own best result
    auto& thread_best = threads_.at(t)->GetStats().thread_best;
    global_best_.UpdateIfBetter(thread_best, optimization_metrics_);
    // std::cout << "Thread " << t << " best layout:" << std::endl;
    // layout::PrintOverallLayoutConcise(global_best_.layout);
  }

  std::cout << std::endl;

  for (unsigned t = 0; t < num_threads_; t++)
  {
    delete threads_.at(t);
    threads_.at(t) = nullptr;
  }

  std::stringstream map_txt_str;
  std::stringstream map_yaml_str;
  std::stringstream map_cpp_str;
  std::stringstream stats_str;
  std::stringstream xml_map_stats_str;
  std::stringstream tensella_str;
  if (global_best_.valid)
  {
    global_best_.mapping.PrettyPrint(map_txt_str, arch_specs_.topology.StorageLevelNames(),
                                    global_best_.stats.utilized_capacities,
                                    global_best_.stats.tile_sizes);

    // std::ofstream map_yaml_file(map_yaml_file_name);
    // global_best_.mapping.PrintAsConstraints(map_yaml_file_name);
    // map_yaml_file.close();

    // Re-evaluate the mapping so that we get a live engine with complete specs and stats
    // that can be printed out hierarchically.
    model::Engine engine;
    engine.Spec(arch_specs_);

    if (layout_initialized_){
      engine.Evaluate(global_best_.mapping, workload_, layout_, sparse_optimizations_, crypto_);
    }else
      engine.Evaluate(global_best_.mapping, workload_, global_best_.layout, sparse_optimizations_, crypto_);

    stats_str << engine << std::endl;

    if (emit_whoop_nest_)
    {
      global_best_.mapping.PrintWhoopNest(map_cpp_str, arch_specs_.topology.StorageLevelNames(),
                                          global_best_.stats.tile_sizes,
                                          global_best_.stats.utilized_instances);
    }

    std::cout << std::endl;
    if (!sparse_optimizations_->no_optimization_applied)
    {
      std::cout << "Summary stats for best mapping found by mapper:" << std::endl;
      std::cout << "  Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2)
                << global_best_.stats.utilization << " | pJ/Algorithmic-Compute = " << std::setw(8)
                << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << global_best_.stats.energy /
        global_best_.stats.algorithmic_computes
                << " | pJ/Compute = " << std::setw(8)
                << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << global_best_.stats.energy /
        global_best_.stats.actual_computes
                << " | Cycles = " << global_best_.stats.cycles << std::endl;
    }
    else
    {
      std::cout << "Summary stats for best mapping found by mapper:" << std::endl;
      std::cout << "  Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2)
                << global_best_.stats.utilization
                << " | pJ/Compute = " << std::setw(8)
                << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << global_best_.stats.energy /
        global_best_.stats.actual_computes
                << " | Cycles = " << global_best_.stats.cycles << std::endl;
    }

    // Print the engine stats and mapping to an XML file
    boost::archive::xml_oarchive ar(xml_map_stats_str);
    ar << boost::serialization::make_nvp("engine", engine);
    ar << boost::serialization::make_nvp("mapping", global_best_.mapping);
    const Mapper* a = this;
    ar << BOOST_SERIALIZATION_NVP(a);

    // Print the mapping in Tenssella input format.
    global_best_.mapping.PrintTenssella(tensella_str);
    layout::PrintOverallLayoutConcise(global_best_.layout);
  }
  else
  {
    std::cout << "MESSAGE: no valid mappings found within search criteria. Some suggestions:" << std::endl;
    std::cout << "(1) Observe each mapper thread's termination message. If it terminated due to" << std::endl
              << "    consecutive failed mappings, it will tell you the number of mappings that" << std::endl
              << "    failed because of a spatial fanout violation and the number that failed" << std::endl
              << "    because of a buffer capacity violation." << std::endl;
    std::cout << "(2) Check your architecture configuration (especially mapspace constraints)." << std::endl
              << "    Try to find the offending constraints that are likely to have caused the" << std::endl
              << "    above violations, and disable those constraints." << std::endl;
    std::cout << "(3) Try other search algorithms, and relax the termination criteria:" << std::endl
              << "    victory_condition, timeout and/or search_size." << std::endl;
    if (!diagnostics_on_)
    {
      std::cout << "(4) Enable mapper's diagnostics (mapper.diagnostics = True) to track and emit " << std::endl
                << "    more information about failed mappings." << std::endl;
    }
  }

  // Create an output cfg starting with the original cfg contents.
  // Create an output yaml that contains the best mapping,
  libconfig::Config config;
  YAML::Emitter yaml_out;

  if (cfg_string_) {
    config.readString(cfg_string_);
    free(cfg_string_);
  }
  libconfig::Setting& root = config.getRoot();

#ifdef EMIT_OPT_AS_CONSTRAINTS
  // Update the mapper constraints.
  libconfig::Setting& mapper = root.lookup("mapper");

  if (mapper.exists("algorithm"))
    mapper["algorithm"] = "exhaustive";
  else
    mapper.add("algorithm", libconfig::Setting::TypeString) = "exhaustive";

  if (mapper.exists("num_threads"))
    mapper["num_threads"] = 1;
  else
    mapper.add("num_threads", libconfig::Setting::TypeInt) = 1;

  if (mapper.exists("search_size"))
    mapper.remove("search_size");

  if (mapper.exists("search_size"))
    mapper["search_size"] = 1;
  else
    mapper.add("search_size", libconfig::Setting::TypeInt) = 1;

  // Delete the mapspace constraint.
  if (root.exists("mapspace"))
    root.remove("mapspace");

  if (global_best_.valid)
  {
    // Create a new mapspace constraint.
    libconfig::Setting& mapspace = root.add("mapspace", libconfig::Setting::TypeGroup);

    // Format the best mapping as libconfig constraints.
    global_best_.mapping.FormatAsConstraints(mapspace);
  }
#else
  // We used to create a set of mapper constraints to fit exactly one mapping,
  // which could then be provided to timeloop-mapper.
  // We now create a single mapping which can be fed to timeloop-model.
  if (root.exists("mapper"))
    root.remove("mapper");

  if (root.exists("mapspace"))
    root.remove("mapspace");

  if (global_best_.valid)
  {
    // Create a new mapping.
    libconfig::Setting& mapping = root.add("mapping", libconfig::Setting::TypeList);

    // Format the best mapping as a libconfig spec.
    global_best_.mapping.FormatAsLibConfig(mapping, arch_specs_.topology.StorageLevelNames());

    yaml_out << YAML::BeginMap;
    yaml_out << YAML::Key << "mapping";
    yaml_out << YAML::Value;
    yaml_out << YAML::BeginSeq;
    global_best_.mapping.FormatAsYaml(yaml_out, arch_specs_.topology.StorageLevelNames());
    yaml_out << YAML::EndSeq;
    yaml_out << YAML::EndMap;

    // Dump the global best layout to YAML file
    std::string layout_filename = out_prefix_ + ".layout.yaml";
    layout::DumpLayoutToYAML(global_best_.layout, layout_filename);
    std::cout << "Best layout saved to " << layout_filename << std::endl;
  }
#endif
  if (!cfg_string_) {
    map_yaml_str << yaml_out.c_str();
  } else {
    config.writeFile(map_cfg_file_name.c_str());
  }

  Result result;
  result.mapping_cpp_string = map_cpp_str.str();
  result.mapping_yaml_string = map_yaml_str.str();
  result.mapping_string = map_txt_str.str();
  result.stats_string = stats_str.str();
  result.tensella_string = tensella_str.str();
  result.xml_mapping_stats_string = xml_map_stats_str.str();
  result.orojenesis_string = orojenesis_stream.str();

  return result;
}

} // namespace application