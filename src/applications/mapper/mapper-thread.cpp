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

#include <ncurses.h>

#include "applications/mapper/mapper-thread.hpp"
#include "layoutspaces/layoutspace.hpp"

// #define DEBUG_SHOW_MAPPING_LAYOUT
// #define DEBUG_SHOW_LAYOUT_SEARCHING
// #define BANDWIDTH_MODEL_MAPPING_SEARCH // uncomment out to use memory bandwidth-based mapping search.
#define LESS_IMPROVEMENT_COUNTER_THRESHOLD 10

bool gTerminate = false;

enum class Betterness
{
  Better,
  SlightlyBetter,
  SlightlyWorse,
  Worse
};

static std::uint64_t SumStats(problem::PerDataSpace<std::uint64_t>& data, problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces)
{
  if (pv != problem::GetShape()->NumDataSpaces)
  {
    return data.at(pv);
  }
  else
  {
    std::uint64_t stat = 0;
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      stat += SumStats(data, problem::Shape::DataSpaceID(pvi));
    }
    return stat;
  }
}

static double Cost(const model::Topology::Stats& stats, const std::string metric)
{
  double cost;
  if (metric == "delay")
  {
    cost = static_cast<double>(stats.cycles);
  }
  else if (metric == "energy")
  {
    cost = stats.energy;
  }
  else if (metric == "last_level_accesses")
  {
    cost = stats.last_level_accesses;
  }
  else if (metric.compare(0, 9, "accesses-") == 0)
  {
    unsigned level = unsigned(atoi(metric.substr(9).c_str()));
    cost = stats.accesses.at(level);
  }
  else
  {
    assert(metric == "edp");
    cost = (stats.energy * stats.cycles);
  }
  return cost;
}

static Betterness IsBetterRecursive_(const model::Topology::Stats& candidate, const model::Topology::Stats& incumbent,
                                     const std::vector<std::string>::const_iterator metric,
                                     const std::vector<std::string>::const_iterator end)
{
  const double tolerance = 0.001;

  double candidate_cost = Cost(candidate, *metric);
  double incumbent_cost = Cost(incumbent, *metric);

  // Compute % improvement relative to incumbent. We need to
  // special-case cost == 0 to avoid a divide-by-zero error. Note that
  // cost == 0 is a legitimate cost for a mapping. Also note that lower
  // cost is better.
  double absolute_improvement = incumbent_cost - candidate_cost;
  double relative_improvement = incumbent_cost == 0 ?
    (candidate_cost == 0 ? 0 : absolute_improvement / candidate_cost) :
    absolute_improvement / incumbent_cost;

  if (fabs(relative_improvement) > tolerance)
  {
    // We have a clear winner.
    if (relative_improvement > 0)
      return Betterness::Better;
    else
      return Betterness::Worse;
  }
  else
  {
    // Within tolerance range, try to recurse.
    if (std::next(metric) == end)
    {
      // Base case. NOTE! Equality is categorized as SlightlyWorse (prefers incumbent).
      if (relative_improvement > 0)
        return Betterness::SlightlyBetter;
      else
        return Betterness::SlightlyWorse;
    }
    else
    {
      // Recursive call.
      Betterness lsm = IsBetterRecursive_(candidate, incumbent, std::next(metric), end);
      if (lsm == Betterness::Better || lsm == Betterness::Worse)
        return lsm;
      // NOTE! Equality is categorized as SlightlyWorse (prefers incumbent).
      else if (relative_improvement > 0)
        return Betterness::SlightlyBetter;
      else
        return Betterness::SlightlyWorse;
    }
  }
}

static inline bool IsBetter(const model::Topology::Stats& candidate, const model::Topology::Stats& incumbent,
                            const std::vector<std::string>& metrics)
{
  Betterness b = IsBetterRecursive_(candidate, incumbent, metrics.begin(), metrics.end());
  return (b == Betterness::Better || b == Betterness::SlightlyBetter);
}

static inline bool IsEqual(const model::Topology::Stats& candidate, const model::Topology::Stats& incumbent,
                            const std::vector<std::string>& metrics)
{
  Betterness b = IsBetterRecursive_(candidate, incumbent, metrics.begin(), metrics.end());
  return (b == Betterness::SlightlyWorse);
}

bool EvaluationResult::UpdateIfBetter(const EvaluationResult& other, const std::vector<std::string>& metrics)
{
  bool updated = false;
  if (other.valid &&
      (!valid || IsBetter(other.stats, stats, metrics)))
  {
    valid = true;
    mapping = other.mapping;
    stats = other.stats;
    layout = other.layout;  // Copy layout
    updated = true;
  }
  return updated;
}

bool EvaluationResult::UpdateIfEqual(const EvaluationResult& other, const std::vector<std::string>& metrics)
{
  bool updated = false;
  if (other.valid &&
      (!valid || IsEqual(other.stats, stats, metrics)))
  {
    valid = true;
    mapping = other.mapping;
    stats = other.stats;
    layout = other.layout;  // Copy layout
    updated = true;
  }
  return updated;
}

//--------------------------------------------//
//              Failure Tracking              //
//--------------------------------------------//

std::map<FailClass, std::string> FailClassToString =
{
  { FailClass::Fanout, "Fanout" },
  { FailClass::Capacity, "Capacity" }
};

std::ostream& operator << (std::ostream& out, const FailClass& fail_class)
{
  out << FailClassToString.at(fail_class);
  return out;
}

//--------------------------------------------//
//               Mapper Thread                //
//--------------------------------------------//

MapperThread::Stats::Stats() :
    distribution(0.0,1.0)
{
}

void MapperThread::Stats::UpdateFails(FailClass fail_class, std::string fail_reason, unsigned level, const Mapping& mapping)
{
  // Find the data corresponding to this fail class.
  auto fail_bucket_it = fail_stats.find(fail_class);
  if (fail_bucket_it == fail_stats.end())
  {
    // We've never seen this fail class before.
    std::map<unsigned, FailInfo> fail_bucket;
    fail_bucket[level].count = 1;
    fail_bucket[level].mapping = mapping;
    fail_bucket[level].reason = fail_reason;
    fail_stats[fail_class] = fail_bucket;
  }
  else
  {
    // We've seen this fail class, see if this level has
    // failed in this class.
    auto& fail_bucket = fail_bucket_it->second;
    auto fail_info_it = fail_bucket.find(level);
    if (fail_info_it == fail_bucket.end())
    {
      // No, this is the first time this level has failed in
      // this fail class, create a new entry.
      fail_bucket[level].count = 1;
      fail_bucket[level].mapping = mapping;
      fail_bucket[level].reason = fail_reason;
    }
    else
    {
      // This level has already failed in this class,
      // increment its count.
      fail_info_it->second.count += 1;

      // p(x) = prob. that I switch to x when it arrives
      // p(0) = 1

      // P(x) = prob. that x is finally selected.
      // 1/N = P(0) = p(0).(1-p(1)).(1-p(2))...(1-p(N-1))
      // 1/N = P(1) =        (p(1)).(1-p(2))...(1-p(N-1))

      // p(x).(1-p(x+1)) = p(x+1)
      // ...
      // => p(x+1) = p(x) / [1+p(x)]
      // ...
      // => p(x) = 1/(1+x)

      // Compute the probability of switching (we've already computed count=x+1)
      double prob = 1 / fail_info_it->second.count.convert_to<double>();

      // Probabilistically update the mapping.
      double roll = distribution(generator);
      if (roll < prob)
      {
        fail_info_it->second.mapping = mapping;
        fail_info_it->second.reason = fail_reason;
      }
    }
  }
}


MapperThread::MapperThread(
  unsigned thread_id,
  search::SearchAlgorithm* search,
  mapspace::MapSpace* mapspace,
  std::mutex* mutex,
  uint128_t search_size,
  std::uint32_t timeout,
  std::uint32_t victory_condition,
  std::int32_t max_temporal_loops_in_a_mapping,
  uint128_t sync_interval,
  uint128_t log_interval,
  bool log_orojenesis_mappings,
  bool log_mappings_yaml,
  bool log_mappings_verbose,
  bool log_all_mappings,
  bool log_stats,
  bool log_suboptimal,
  std::ostream& log_stream,
  std::ostream& orojenesis_csv_file,
  std::string orojenesis_prefix,
  bool live_status,
  bool diagnostics_on,
  bool penalize_consecutive_bypass_fails,
  std::vector<std::string> optimization_metrics,
  model::Engine::Specs arch_specs,
  problem::Workload &workload,
  layout::Layouts layout,
  bool layout_initialized,
  sparse::SparseOptimizationInfo* sparse_optimizations,
  crypto::CryptoConfig* crypto,
  EvaluationResult* best
  ):
    thread_id_(thread_id),
    search_(search),
    mapspace_(mapspace),
    mutex_(mutex),
    search_size_(search_size),
    timeout_(timeout),
    victory_condition_(victory_condition),
    max_temporal_loops_in_a_mapping_(max_temporal_loops_in_a_mapping),
    sync_interval_(sync_interval),
    log_interval_(log_interval),
    log_orojenesis_mappings_(log_orojenesis_mappings),
    log_all_mappings_(log_all_mappings),
    log_mappings_yaml_(log_mappings_yaml),
    log_mappings_verbose_(log_mappings_verbose),
    log_stats_(log_stats),
    log_suboptimal_(log_suboptimal),
    log_stream_(log_stream),
    orojenesis_csv_file_(orojenesis_csv_file),
    orojenesis_prefix_(orojenesis_prefix),
    live_status_(live_status),
    diagnostics_on_(diagnostics_on),
    penalize_consecutive_bypass_fails_(penalize_consecutive_bypass_fails),
    optimization_metrics_(optimization_metrics),
    arch_specs_(arch_specs),
    workload_(workload),
    layout_(layout),
    layout_initialized_(layout_initialized),
    sparse_optimizations_(sparse_optimizations),
    crypto_(crypto),
    best_(best),
    thread_(),
    stats_()
{
}

void MapperThread::Start()
{
  // We can do this because std::thread is movable.
  thread_ = std::thread(&MapperThread::Run, this);
}

void MapperThread::Join()
{
  thread_.join();
}

const MapperThread::Stats& MapperThread::GetStats() const
{
  return stats_;
}

void MapperThread::Run()
{
  uint128_t total_mappings = 0;
  uint128_t valid_mappings = 0;
  uint128_t invalid_mappings_mapcnstr = 0;
  uint128_t invalid_mappings_eval = 0;
  std::uint32_t mappings_since_last_best_update = 0;

  const int ncurses_line_offset = 6;

  std::vector<EvaluationResult> index_factor_best_vec;
  model::Engine engine;
  engine.Spec(arch_specs_);

  mapspace::ID prev_mapping_id;

  // =================
  // Main mapper loop -- search mapping
  // =================
  while (true)
  {
    if (live_status_)
    {
      std::stringstream msg;

      msg << std::setw(3) << thread_id_ << std::setw(11) << total_mappings
          << std::setw(11) << (total_mappings - valid_mappings)  << std::setw(11) << valid_mappings
          << std::setw(11) << invalid_mappings_mapcnstr + invalid_mappings_eval
          << std::setw(11) << mappings_since_last_best_update;

      if (valid_mappings > 0)
      {
        msg << std::setw(10) << OUT_FLOAT_FORMAT << std::setprecision(2) << OUT_PERCENT(stats_.thread_best.stats.utilization)
            << std::setw(11) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats_.thread_best.stats.energy /
          stats_.thread_best.stats.algorithmic_computes
            << std::setw(11) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats_.thread_best.stats.cycles;
        msg << std::endl;
      }

      mutex_->lock();
      mvaddstr(thread_id_ + ncurses_line_offset, 0, msg.str().c_str());

      refresh();
      mutex_->unlock();
    }

    // Termination conditions.
    bool terminate = false;

    if (gTerminate)
    {
      mutex_->lock();
      log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: "
                  << "global termination flag activated, terminating search."
                  << std::endl;
      mutex_->unlock();
      terminate = true;
    }

    if (search_size_ > 0 && valid_mappings >= search_size_)
    {
      mutex_->lock();
      log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: " << search_size_
                  << " valid mappings found, terminating search."
                  << std::endl;
      mutex_->unlock();
      terminate = true;
    }

    if (victory_condition_ > 0 && mappings_since_last_best_update >= victory_condition_)
    {
      mutex_->lock();
      log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: " << victory_condition_
                  << " suboptimal mappings found since the last upgrade, terminating search."
                  << std::endl;
      mutex_->unlock();
      terminate = true;
    }

    if ((invalid_mappings_mapcnstr + invalid_mappings_eval) > 0 &&
        (invalid_mappings_mapcnstr + invalid_mappings_eval) >= timeout_)
    {
      mutex_->lock();
      log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: " << timeout_
                  << " invalid mappings (" << invalid_mappings_mapcnstr << " fanout, "
                  << invalid_mappings_eval << " capacity) found since the last valid mapping, "
                  << "terminating search." << std::endl;
      mutex_->unlock();
      terminate = true;
    }

    // Try to obtain the next mapping from the search algorithm.
    mapspace::ID mapping_id;
    if (!search_->Next(mapping_id))
    {
      mutex_->lock();
      log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: "
                  << "search algorithm is done, terminating search."
                  << std::endl;
      mutex_->unlock();
      terminate = true;
    }

    if ((log_orojenesis_mappings_ || log_all_mappings_) && terminate)
    {
      for (auto &index_factor_best : index_factor_best_vec)
      {
        // Re-evaluate the mapping
        if (layout_initialized_){
          engine.Evaluate(index_factor_best.mapping, workload_, layout_, sparse_optimizations_, crypto_, !diagnostics_on_);
        }else
          engine.Evaluate(index_factor_best.mapping, workload_, layout_, sparse_optimizations_, crypto_, !diagnostics_on_);

        if (index_factor_best.valid) {
            auto topology = engine.GetTopology();
            mutex_->lock();
            // Print performance and log the optimal mappings
            topology.PrintOrojenesis(&workload_, orojenesis_csv_file_, index_factor_best.mapping, log_mappings_yaml_, log_mappings_verbose_, orojenesis_prefix_, thread_id_);
            mutex_->unlock();
        }
      }

      // Reset the best for next permutation/bypassing
      stats_.index_factor_best.valid = false;
      index_factor_best_vec.clear();
    }

    // Terminate.
    if (terminate)
    {
      if (live_status_)
      {
        mutex_->lock();
        mvaddstr(thread_id_ + ncurses_line_offset, 0, "-");
        refresh();
        mutex_->unlock();
      }

      std::cout << "done mapping search" << std::endl;
      std::cout << "optimal mapping: " << stats_.thread_best.mapping << std::endl;
      std::cout << "stats_.thread_best.valid: " << stats_.thread_best.valid << std::endl;
      std::cout << "layout_initialized_: " << layout_initialized_ << std::endl;
      // =================
      // Secondary mapper loop -- search layout
      // =================
      // Perform layout search on the final best mapping (only if layout was not pre-initialized)
      if (!layout_initialized_ && stats_.thread_best.valid)
      {
        #ifdef DEBUG_SHOW_LAYOUT_SEARCHING
        log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: "
                    << "Performing layout optimization on best mapping found."
                    << std::endl;
        #endif

        layoutspace_ = new layoutspace::Legal(arch_specs_, stats_.thread_best.mapping, layout_);

        // --- Add AuthBlock nest with dummy values for DRAM and MainMemory ---
        for(unsigned lvl = 0; lvl < layout_.size(); lvl++){
          for (const auto &ds : layout_[lvl].data_space){
            if (layout_[lvl].target == "DRAM" or layout_[lvl].target == "MainMemory"){
              layout::LayoutNest authblock_nest;
              authblock_nest.data_space = ds;
              authblock_nest.type = "authblock_lines";
              authblock_nest.ranks = layout_[lvl].dataSpaceToRank[ds];
              // Set all factors to 1 for dummy layout
              for (const auto &r : authblock_nest.ranks)
              {
                authblock_nest.factors[r] = 1;
              }
              layout_[lvl].authblock_lines.push_back(authblock_nest);
            }
          }
        }

        layoutspace_->Init(arch_specs_, stats_.thread_best.mapping, layout_, false); // need the layout for architecture information.
        auto concordant_layout = layoutspace_->GetLayout();
        // Initialize global optimal tracking variables
        std::uint64_t mapping_specific_best_latency = UINT64_MAX;
        double mapping_specific_best_energy_per_compute = std::numeric_limits<double>::max();
        layout::Layouts mapping_specific_best_layout;
        bool has_valid_layout = false;

        log_stream_ << "[" << thread_id_ << "] Starting independent ordered layout optimization:" << std::endl;
        log_stream_ << "[" << thread_id_ << "] - " <<  layoutspace_->splitting_candidates << " SplittingSpace candidates" << std::endl;
        log_stream_ << "[" << thread_id_ << "] - " << layoutspace_->packing_candidates << " PackingSpace candidates" << std::endl;
        log_stream_ << "[" << thread_id_ << "] - " << layoutspace_->authblock_candidates << " AuthSpace candidates" << std::endl;

        // Track best IDs for each design space
        uint64_t best_layout_splitting_id = 0;
        uint64_t best_layout_packing_id = 0;
        uint64_t best_layout_auth_id = 0;

        // Phase 1: Search SplittingSpace (with cleared authblock_lines and default PackingSpace=0)
        log_stream_ << "[" << thread_id_ << "] Phase 1: Optimizing SplittingSpace (clearing authblock_lines for pure evaluation)..." << std::endl;
        for (uint64_t layout_splitting_id = 0; layout_splitting_id < layoutspace_->splitting_candidates; layout_splitting_id++)
        {
          #ifdef DEBUG_SHOW_LAYOUT_SEARCHING
            log_stream_ << "[" << thread_id_ << "] Testing SplittingSpace " << layout_splitting_id << "/" << layoutspace_->splitting_candidates << std::endl;
          #endif 
          auto construction_status = layoutspace_->ConstructLayout(layout_splitting_id, 0, 0, &layout_, stats_.thread_best.mapping, false);
          bool layout_success = std::accumulate(construction_status.begin(), construction_status.end(), true,
                                      [](bool cur, const layoutspace::Status& status)
                                      { return cur && status.success; });
          if(!layout_success) {
            log_stream_ << "[" << thread_id_ << "] SplittingSpace " << layout_splitting_id << " construction failed -- reason:" << construction_status[0].fail_reason << std::endl;
            continue;
          }

          #ifdef DEBUG_SHOW_LAYOUT_SEARCHING
            log_stream_ << "[" << thread_id_ << "] Cleared authblock_lines for PackingSpace " << layout_packing_id << " evaluation" << std::endl;
          #endif 
          layout::Layouts layout_no_auth = layout::Layouts(layout_);
          for (unsigned lvl = 0; lvl < layout_no_auth.size(); lvl++) {
            for (unsigned ds_idx = 0; ds_idx < layout_no_auth[lvl].authblock_lines.size(); ds_idx++) {
              // Clear all authblock factors to eliminate their effect
              layout_no_auth[lvl].authblock_lines[ds_idx].factors.clear();
            }
          }

          auto status_per_level = engine.Evaluate(stats_.thread_best.mapping, workload_, layout_no_auth, sparse_optimizations_, crypto_, !diagnostics_on_);

          // Extract run-time latency and energy efficiency from evaluation results
          std::uint64_t runtime_latency = engine.Cycles();
          double total_energy = engine.Energy();
          std::uint64_t actual_computes = engine.GetTopology().ActualComputes();
          double energy_per_compute = (actual_computes > 0) ? (total_energy / actual_computes) : 0.0;

          // Track optimal values (prefer smaller values)
          bool is_better = false;
          std::string improvement_reason = "";

          if (!has_valid_layout) {
            // First valid layout
            is_better = true;
            improvement_reason = "SplittingSpace: first valid layout";
          }
          else if (runtime_latency < mapping_specific_best_latency) {
            // Better latency
            is_better = true;
            improvement_reason = "SplittingSpace: better latency";
          }
          else if (runtime_latency == mapping_specific_best_latency && energy_per_compute < mapping_specific_best_energy_per_compute) {
            // Same latency but better energy efficiency
            is_better = true;
            improvement_reason = "SplittingSpace: same latency, better energy efficiency";
          }

          if (is_better) {
            mapping_specific_best_latency = runtime_latency;
            mapping_specific_best_energy_per_compute = energy_per_compute;
            mapping_specific_best_layout = layout_; // Store the layout with cleared authblock_lines
            best_layout_splitting_id = layout_splitting_id;
            has_valid_layout = true;

            log_stream_ << "[" << thread_id_ << "] NEW INTRALINE OPTIMAL: ID=" << best_layout_splitting_id
                        << ", Latency=" << mapping_specific_best_latency << " cycles"
                        << ", Energy/Compute=" << mapping_specific_best_energy_per_compute << " pJ/compute"
                        << " (" << improvement_reason << ") [authblock_lines cleared]" << std::endl;
          }
        }

        // Phase 2: Search PackingSpace (with best SplittingSpace and default AuthSpace=0)
        // Note: authblock_lines clearing from Phase 1 does not affect this phase as layout is reconstructed
        if (layoutspace_->packing_candidates > 1) {
#ifdef DEBUG_SHOW_LAYOUT_SEARCHING
          log_stream_ << "[" << thread_id_ << "] Phase 2: Optimizing PackingSpace with best SplittingSpace=" << best_layout_splitting_id << " (authblock_lines restored)..." << std::endl;
#endif 

          for (uint64_t layout_packing_id = 0; layout_packing_id < layoutspace_->packing_candidates; layout_packing_id++)
          {
#ifdef DEBUG_SHOW_LAYOUT_SEARCHING
            log_stream_ << "[" << thread_id_ << "] Testing PackingSpace " << layout_packing_id << "/" << layoutspace_->packing_candidates << std::endl;
#endif
            auto construction_status = layoutspace_->ConstructLayout(best_layout_splitting_id, layout_packing_id, 0,  &layout_, stats_.thread_best.mapping, false);
            bool layout_success = std::accumulate(construction_status.begin(), construction_status.end(), true,
                                        [](bool cur, const layoutspace::Status& status)
                                        { return cur && status.success; });
            if(!layout_success) {
              log_stream_ << "[" << thread_id_ << "] PackingSpace " << layout_packing_id << " construction failed -- reason:" << construction_status[0].fail_reason << std::endl;
              continue;
            }

            layout::Layouts layout_no_auth = layout::Layouts(layout_);
            for (unsigned lvl = 0; lvl < layout_no_auth.size(); lvl++) {
              for (unsigned ds_idx = 0; ds_idx < layout_no_auth[lvl].authblock_lines.size(); ds_idx++) {
                // Clear all authblock factors to eliminate their effect
                layout_no_auth[lvl].authblock_lines[ds_idx].factors.clear();
              }
            }

            auto status_per_level = engine.Evaluate(stats_.thread_best.mapping, workload_, layout_no_auth, sparse_optimizations_, crypto_, !diagnostics_on_);

            // Extract run-time latency and energy efficiency from evaluation results
            std::uint64_t runtime_latency = engine.Cycles();
            double total_energy = engine.Energy();
            std::uint64_t actual_computes = engine.GetTopology().ActualComputes();
            double energy_per_compute = (actual_computes > 0) ? (total_energy / actual_computes) : 0.0;

            // Check if better than current best
            bool is_better = false;
            std::string improvement_reason = "";

            if (runtime_latency < mapping_specific_best_latency) {
              is_better = true;
              improvement_reason = "PackingSpace: better latency";
            }
            else if (runtime_latency == mapping_specific_best_latency && energy_per_compute < mapping_specific_best_energy_per_compute) {
              is_better = true;
              improvement_reason = "PackingSpace: same latency, better energy efficiency";
            }

            if (is_better) {
              mapping_specific_best_latency = runtime_latency;
              mapping_specific_best_energy_per_compute = energy_per_compute;
              mapping_specific_best_layout = layout_;
              best_layout_packing_id = layout_packing_id;
              has_valid_layout = true;

              log_stream_ << "[" << thread_id_ << "] NEW PACKING OPTIMAL: ID=" << best_layout_packing_id
                          << ", Latency=" << mapping_specific_best_latency << " cycles"
                          << ", Energy/Compute=" << mapping_specific_best_energy_per_compute << " pJ/compute"
                          << " (" << improvement_reason << ")" << std::endl;
            }
          }
        }

        // Phase 3: Search AuthSpace (with best SplittingSpace and best PackingSpace)
        // Note: authblock_lines are fully functional in this phase
        if (layoutspace_->authblock_candidates > 1) {
#ifdef DEBUG_SHOW_LAYOUT_SEARCHING
          log_stream_ << "[" << thread_id_ << "] Phase 3: Optimizing AuthSpace with best SplittingSpace=" << best_layout_splitting_id
                      << " and PackingSpace=" << best_layout_packing_id << " (authblock_lines active)..." << std::endl;
#endif 
          log_stream_ << "[" << thread_id_ << "] Add AuthBlock and reinitializes best efficiency and latency" << std::endl;
          mapping_specific_best_latency = UINT64_MAX;
          mapping_specific_best_energy_per_compute = std::numeric_limits<double>::max();
          uint32_t less_improvement_counter = 0;

          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_int_distribution<uint64_t> dist(0, layoutspace_->authblock_candidates - 1);
          for (uint64_t i = 0; i < layoutspace_->authblock_candidates; i++)
          {
            uint64_t layout_auth_id = dist(gen);
#ifdef DEBUG_SHOW_LAYOUT_SEARCHING
            log_stream_ << "[" << thread_id_ << "] Testing AuthSpace " << layout_auth_id << "/" << layoutspace_->authblock_candidates << std::endl;
#endif 
            auto construction_status = layoutspace_->ConstructLayout(best_layout_splitting_id, best_layout_packing_id, layout_auth_id, &layout_, stats_.thread_best.mapping, false);
            bool layout_success = std::accumulate(construction_status.begin(), construction_status.end(), true,
                                        [](bool cur, const layoutspace::Status& status)
                                        { return cur && status.success; });
            if(!layout_success) {
              log_stream_ << "[" << thread_id_ << "] AuthSpace " << layout_auth_id << " construction failed -- reason:" << construction_status[0].fail_reason << std::endl;
              continue;
            }

            auto status_per_level = engine.Evaluate(stats_.thread_best.mapping, workload_, layout_, sparse_optimizations_, crypto_, !diagnostics_on_);

            // Extract run-time latency and energy efficiency from evaluation results
            std::uint64_t runtime_latency = engine.Cycles();
            double total_energy = engine.Energy();
            std::uint64_t actual_computes = engine.GetTopology().ActualComputes();
            double energy_per_compute = (actual_computes > 0) ? (total_energy / actual_computes) : 0.0;

            // Check if better than current best
            bool is_better = false;
            std::string improvement_reason = "";

            if (runtime_latency < mapping_specific_best_latency) {
              is_better = true;
              improvement_reason = "AuthSpace: better latency";
              less_improvement_counter = 0;
            }
            else if (runtime_latency == mapping_specific_best_latency && energy_per_compute < mapping_specific_best_energy_per_compute) {
              is_better = true;
              improvement_reason = "AuthSpace: same latency, better energy efficiency";
              if ((mapping_specific_best_energy_per_compute - energy_per_compute) < 0.1) {
                less_improvement_counter++;
              }
            }

            if (is_better) {
              mapping_specific_best_latency = runtime_latency;
              mapping_specific_best_energy_per_compute = energy_per_compute;
              mapping_specific_best_layout = layout_;
              best_layout_auth_id = layout_auth_id;

              log_stream_ << "[" << thread_id_ << "] NEW AUTH OPTIMAL: ID=" << best_layout_auth_id
                          << ", Latency=" << mapping_specific_best_latency << " cycles"
                          << ", Energy/Compute=" << mapping_specific_best_energy_per_compute << " pJ/compute"
                          << " (" << improvement_reason << ")" << std::endl;
            }

            if (less_improvement_counter > LESS_IMPROVEMENT_COUNTER_THRESHOLD) {
              break;
            }
          }
        }

        // Final optimal configuration
        log_stream_ << "[" << thread_id_ << "] FINAL OPTIMAL CONFIGURATION: SplittingSpace=" << best_layout_splitting_id
                    << ", PackingSpace=" << best_layout_packing_id << ", AuthSpace=" << best_layout_auth_id
                    << " (authblock_lines effects included in final result)" << std::endl;

        // Update the best result with the optimal layout
        if (has_valid_layout) {
          log_stream_ << "[" << thread_id_ << "] FINAL LAYOUT OPTIMAL: "
          << "Latency=" << mapping_specific_best_latency << " cycles, "
          << "Energy/Compute=" << mapping_specific_best_energy_per_compute << " pJ/compute"
          << std::endl;

          // Print layout details for the optimal layout
          std::cout << "[" << thread_id_ << "] OPTIMAL LAYOUT:" << std::endl;

          layout::PrintOverallLayout(mapping_specific_best_layout);

          // Update the thread best with the optimal layout and re-evaluate to get final stats
          layout_ = mapping_specific_best_layout;
          auto final_status = engine.Evaluate(stats_.thread_best.mapping, workload_, layout_, sparse_optimizations_, crypto_, !diagnostics_on_);
          auto final_topology = engine.GetTopology();
          auto final_stats = final_topology.GetStats();

          // Update the thread best with the final optimized result
          EvaluationResult final_result = { true, stats_.thread_best.mapping, final_stats, layout_ };
          stats_.thread_best = final_result;

          // Sync the final optimized result to global best
          mutex_->lock();
          best_->UpdateIfBetter(stats_.thread_best, optimization_metrics_);
          mutex_->unlock();

        } else {
          log_stream_ << "[" << thread_id_ << "] No valid layouts found for best mapping or no valid design choice in layout, fall back to concordant layout " << std::endl;
          layoutspace_->SequentialFactorizeLayout(concordant_layout);
          auto final_status = engine.Evaluate(stats_.thread_best.mapping, workload_, concordant_layout, sparse_optimizations_, crypto_, !diagnostics_on_);
          auto final_topology = engine.GetTopology();
          auto final_stats = final_topology.GetStats();
          EvaluationResult final_result = { true, stats_.thread_best.mapping, final_stats, concordant_layout };
          stats_.thread_best = final_result;

          // Sync the fallback result to global best
          mutex_->lock();
          best_->UpdateIfBetter(stats_.thread_best, optimization_metrics_);
          mutex_->unlock();
        }
      }
      break;
    }

    //
    // Periodically sync thread_best with global best.
    //
    if (total_mappings != 0 && sync_interval_ > 0 && total_mappings % sync_interval_ == 0)
    {
      mutex_->lock();

      // Sync from global best to thread_best.
      bool global_pulled = false;
      if (best_->valid)
      {
        if (stats_.thread_best.UpdateIfBetter(*best_, optimization_metrics_))
        {
          global_pulled = true;
        }
      }

      // Sync from thread_best to global best.
      if (stats_.thread_best.valid && !global_pulled)
      {
        best_->UpdateIfBetter(stats_.thread_best, optimization_metrics_);
      }

      mutex_->unlock();
    }

    //
    // Check if the only change vs. the previous mapping was in the Bypass
    // dimension. This is useful later.
    //
    bool only_bypass_changed = false;
    if (total_mappings > 1)
    {
      // Match ON if the bypass changed
      for (unsigned idim = 0; idim < unsigned(mapspace::Dimension::Num); idim++)
      {
        if (mapspace::Dimension(idim) == mapspace::Dimension::DatatypeBypass)
          only_bypass_changed |= (mapping_id[idim] != prev_mapping_id[idim]);
      }
      // OFF if anything else changed
      for (unsigned idim = 0; idim < unsigned(mapspace::Dimension::Num); idim++)
      {
        if (mapspace::Dimension(idim) != mapspace::Dimension::DatatypeBypass)
          only_bypass_changed &= (mapping_id[idim] == prev_mapping_id[idim]);
      }
    }
    prev_mapping_id = mapping_id;

    //
    // Begin Mapping. We do this in several stages with increasing algorithmic
    // complexity and attempt to bail out as quickly as possible at each stage.
    //
    bool success = true;

    // Stage 1: Construct a mapping from the mapping ID. This step can fail
    //          because the space of *legal* mappings isn't dense (unfortunately),
    //          so a mapping ID may point to an illegal mapping.
    Mapping mapping(&workload_);

    auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
    success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                               [](bool cur, const mapspace::Status& status)
                               { return cur && status.success; });

    total_mappings++;
    if(success && max_temporal_loops_in_a_mapping_ > 0)
    { // Count the number of temporal loops
      int temporal_loops = 0;
      for(auto& maploop: mapping.loop_nest.loops)
      {
        if(loop::IsSpatial(maploop.spacetime_dimension)) continue;
        temporal_loops += (maploop.end - maploop.start) > maploop.stride;
      }
      if(temporal_loops > max_temporal_loops_in_a_mapping_) success = false;
    }


    if (!success)
    {
      invalid_mappings_mapcnstr++;
      if (diagnostics_on_)
      {
        for (unsigned level = 0; level < construction_status.size(); level++)
          if (!construction_status.at(level).success)
            stats_.UpdateFails(FailClass::Fanout, construction_status.at(level).fail_reason, level, mapping);
      }
      search_->Report(search::Status::MappingConstructionFailure);
      continue;
    }

    // Stage 2: (Re)Configure a hardware model to evaluate the mapping
    //          on, and run some lightweight pre-checks that the
    //          model can use to quickly reject a nest.
    //engine.Spec(arch_specs_);
    auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
    success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus& status)
                               { return cur && status.success; });
    layout::Layouts layout_local;
    if (!success)
    {
      // Pre-evaluation failed.
      // If the only change in this mapping vs. the previous mapping was in
      // its dataspace bypass scheme, then we may not want to make this
      // failure count towards the timeout termination trigger.
      if (penalize_consecutive_bypass_fails_ || !only_bypass_changed)
      {
        invalid_mappings_eval++;
      }

      if (diagnostics_on_)
      {
        for (unsigned level = 0; level < status_per_level.size(); level++)
          if (!status_per_level.at(level).success)
            stats_.UpdateFails(FailClass::Capacity, status_per_level.at(level).fail_reason, level, mapping);
      }
      search_->Report(search::Status::EvalFailure);
      continue;
    }

    // Stage 3: Heavyweight evaluation.
    if (layout_initialized_){ // ToDo: @Jianming modify here
      status_per_level = engine.Evaluate(mapping, workload_, layout_, sparse_optimizations_, crypto_, !diagnostics_on_);
      success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus& status)
                               { return cur && status.success; });
    }else{
      // When layout is not initialized, just using bandwidth layout to search the mapping first.

      #ifdef BANDWIDTH_MODEL_MAPPING_SEARCH
        status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);

        if (live_status_)
        {
          std::stringstream msg;

          msg << std::setw(3) << thread_id_ << std::setw(11) << total_mappings
              << std::setw(11) << (total_mappings - valid_mappings)  << std::setw(11) << valid_mappings
              << std::setw(11) << invalid_mappings_mapcnstr + invalid_mappings_eval
              << std::setw(11) << mappings_since_last_best_update;

          if (valid_mappings > 0)
          {
            msg << std::setw(10) << OUT_FLOAT_FORMAT << std::setprecision(2) << OUT_PERCENT(stats_.thread_best.stats.utilization)
                << std::setw(11) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats_.thread_best.stats.energy /
              stats_.thread_best.stats.algorithmic_computes
                << std::setw(11) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats_.thread_best.stats.cycles;
            msg << std::endl;
            msg << "Mapping: " << mapping << std::endl;
            layout::PrintOverallLayoutConcise(layout_local, msg);
            msg << std::endl;
          }

          mutex_->lock();
          mvaddstr(thread_id_ + ncurses_line_offset, 0, msg.str().c_str());

          refresh();
          mutex_->unlock();
        }


        success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                [](bool cur, const model::EvalStatus& status)
                                { return cur && status.success; });
                                
      #endif

      layoutspace_ = new layoutspace::Legal(arch_specs_, mapping, layout_);
      layoutspace_->Init(arch_specs_, mapping, layout_, true);
      auto concordant_layout = layoutspace_->GetLayout();
      // Initialize global optimal tracking variables
      std::uint64_t mapping_specific_best_latency = UINT64_MAX;
      double mapping_specific_best_energy_per_compute = std::numeric_limits<double>::max();
      layout::Layouts mapping_specific_best_layout;
      bool has_valid_layout = false;

      #ifdef DEBUG_SHOW_LAYOUT_SEARCHING
      log_stream_ << "[" << thread_id_ << "] Starting independent ordered layout optimization:" << std::endl;
      log_stream_ << "[" << thread_id_ << "] - " <<  layoutspace_->splitting_candidates << " SplittingSpace candidates" << std::endl;
      log_stream_ << "[" << thread_id_ << "] - " << layoutspace_->packing_candidates << " PackingSpace candidates" << std::endl;
      #endif

      // Track best IDs for each design space
      uint64_t local_best_layout_splitting_id = 0;

      // Phase 1: Search SplittingSpace (with cleared authblock_lines and default PackingSpace=0)
      #ifdef DEBUG_SHOW_LAYOUT_SEARCHING
      log_stream_ << "[" << thread_id_ << "] Phase 1: Optimizing SplittingSpace (clearing authblock_lines for pure evaluation)..." << std::endl;
      #endif
      for (uint64_t layout_splitting_id = 0; layout_splitting_id < layoutspace_->splitting_candidates; layout_splitting_id++)
      {
#ifdef DEBUG_SHOW_LAYOUT_SEARCHING
        log_stream_ << "[" << thread_id_ << "] Testing SplittingSpace " << layout_splitting_id << "/" << layoutspace_->splitting_candidates << std::endl;
#endif 
        auto construction_status = layoutspace_->ConstructLayout(layout_splitting_id, 0, 0, &layout_, mapping, false);
        bool layout_success = std::accumulate(construction_status.begin(), construction_status.end(), true,
                                    [](bool cur, const layoutspace::Status& status)
                                    { return cur && status.success; });
        if(!layout_success) {
          #ifdef DEBUG_SHOW_LAYOUT_SEARCHING
          log_stream_ << "[" << thread_id_ << "] SplittingSpace " << layout_splitting_id << " construction failed -- reason:" << construction_status[0].fail_reason << std::endl;
          #endif
          continue;
        }

        engine.Evaluate(mapping, workload_, layout_, sparse_optimizations_, crypto_, !diagnostics_on_);

        // Extract run-time latency and energy efficiency from evaluation results
        std::uint64_t runtime_latency = engine.Cycles();
        double total_energy = engine.Energy();
        std::uint64_t actual_computes = engine.GetTopology().ActualComputes();
        double energy_per_compute = (actual_computes > 0) ? (total_energy / actual_computes) : 0.0;

        // Track optimal values (prefer smaller values)
        bool is_better = false;
        std::string improvement_reason = "";

        if (!has_valid_layout) {
          // First valid layout
          is_better = true;
          improvement_reason = "SplittingSpace: first valid layout";
        }
        else if (runtime_latency < mapping_specific_best_latency) {
          // Better latency
          is_better = true;
          improvement_reason = "SplittingSpace: better latency";
        }
        else if (runtime_latency == mapping_specific_best_latency && energy_per_compute < mapping_specific_best_energy_per_compute) {
          // Same latency but better energy efficiency
          is_better = true;
          improvement_reason = "SplittingSpace: same latency, better energy efficiency";
        }

        if (is_better) {
          mapping_specific_best_latency = runtime_latency;
          mapping_specific_best_energy_per_compute = energy_per_compute;
          mapping_specific_best_layout = layout_; // Store the layout with cleared authblock_lines
          local_best_layout_splitting_id = layout_splitting_id;
          has_valid_layout = true;

          #ifdef DEBUG_SHOW_LAYOUT_SEARCHING
          log_stream_ << "[" << thread_id_ << "] NEW INTRALINE OPTIMAL: ID=" << local_best_layout_splitting_id
                      << ", Latency=" << mapping_specific_best_latency << " cycles"
                      << ", Energy/Compute=" << mapping_specific_best_energy_per_compute << " pJ/compute"
                      << " (" << improvement_reason << ") [authblock_lines cleared]" << std::endl;
          #endif
        }
      }

      // Phase 2: Search PackingSpace (with best SplittingSpace and default AuthSpace=0)
      // Note: authblock_lines clearing from Phase 1 does not affect this phase as layout is reconstructed
      if (layoutspace_->packing_candidates > 1) {
#ifdef DEBUG_SHOW_LAYOUT_SEARCHING
        log_stream_ << "[" << thread_id_ << "] Phase 2: Optimizing PackingSpace with best SplittingSpace=" << local_best_layout_splitting_id << " (authblock_lines restored)..." << std::endl;
#endif 

        for (uint64_t layout_packing_id = 0; layout_packing_id < layoutspace_->packing_candidates; layout_packing_id++)
        {
#ifdef DEBUG_SHOW_LAYOUT_SEARCHING
          log_stream_ << "[" << thread_id_ << "] Testing PackingSpace " << layout_packing_id << "/" << layoutspace_->packing_candidates << std::endl;
#endif
          auto construction_status = layoutspace_->ConstructLayout(local_best_layout_splitting_id, layout_packing_id, 0,  &layout_, mapping, false);
          bool layout_success = std::accumulate(construction_status.begin(), construction_status.end(), true,
                                      [](bool cur, const layoutspace::Status& status)
                                      { return cur && status.success; });
          if(!layout_success) {
#ifdef DEBUG_SHOW_LAYOUT_SEARCHING
            log_stream_ << "[" << thread_id_ << "] PackingSpace " << layout_packing_id << " construction failed -- reason:" << construction_status[0].fail_reason << std::endl;
#endif
            continue;
          }

          status_per_level = engine.Evaluate(mapping, workload_, layout_, sparse_optimizations_, crypto_, !diagnostics_on_);

          // Extract run-time latency and energy efficiency from evaluation results
          std::uint64_t runtime_latency = engine.Cycles();
          double total_energy = engine.Energy();
          std::uint64_t actual_computes = engine.GetTopology().ActualComputes();
          double energy_per_compute = (actual_computes > 0) ? (total_energy / actual_computes) : 0.0;

          // Check if better than current best
          bool is_better = false;
          std::string improvement_reason = "";

          if (runtime_latency < mapping_specific_best_latency) {
            is_better = true;
            improvement_reason = "PackingSpace: better latency";
          }
          else if (runtime_latency == mapping_specific_best_latency && energy_per_compute < mapping_specific_best_energy_per_compute) {
            is_better = true;
            improvement_reason = "PackingSpace: same latency, better energy efficiency";
          }

          if (is_better) {
            mapping_specific_best_latency = runtime_latency;
            mapping_specific_best_energy_per_compute = energy_per_compute;
            mapping_specific_best_layout = layout_;
            has_valid_layout = true;
            
#ifdef DEBUG_SHOW_LAYOUT_SEARCHING
            log_stream_ << "[" << thread_id_ << "] NEW PACKING OPTIMAL (No AuthBlock): ID=" << layout_packing_id
                        << ", Latency=" << mapping_specific_best_latency << " cycles"
                        << ", Energy/Compute=" << mapping_specific_best_energy_per_compute << " pJ/compute"
                        << " (" << improvement_reason << ")" << std::endl;
#endif
          }
        }
      }

      // Update the best result with the optimal layout
      if (has_valid_layout) {
#ifdef DEBUG_SHOW_LAYOUT_SEARCHING        
        log_stream_ << "[" << thread_id_ << "] FINAL LAYOUT OPTIMAL (No AuthBlock): "
        << "Latency=" << mapping_specific_best_latency << " cycles, "
        << "Energy/Compute=" << mapping_specific_best_energy_per_compute << " pJ/compute"
        << std::endl;
#endif
        // Update the thread best with the optimal layout and re-evaluate to get final stats
        layout_ = mapping_specific_best_layout;
        status_per_level = engine.Evaluate(mapping, workload_, layout_, sparse_optimizations_, crypto_, !diagnostics_on_);
        success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                 [](bool cur, const model::EvalStatus& status)
                                 { return cur && status.success; });

      } else {
#ifdef DEBUG_SHOW_LAYOUT_SEARCHING
        log_stream_ << "[" << thread_id_ << "] No valid layouts found for best mapping or no valid design choice in layout, fall back to concordant layout " << std::endl;
#endif
        layoutspace_->SequentialFactorizeLayout(concordant_layout);
        status_per_level = engine.Evaluate(mapping, workload_, concordant_layout, sparse_optimizations_, crypto_, !diagnostics_on_);
        success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                 [](bool cur, const model::EvalStatus& status)
                                 { return cur && status.success; });
      }
    }

    if (!success)
    {
      // Evaluation failed.
      // If the only change in this mapping vs. the previous mapping was in
      // its dataspace bypass scheme, then we may not want to make this
      // failure count towards the timeout termination trigger.
      if (penalize_consecutive_bypass_fails_ || !only_bypass_changed)
      {
        invalid_mappings_eval++;
      }

      if (diagnostics_on_)
      {
        for (unsigned level = 0; level < status_per_level.size(); level++)
          if (!status_per_level.at(level).success)
            stats_.UpdateFails(FailClass::Capacity, status_per_level.at(level).fail_reason, level, mapping);
      }
      search_->Report(search::Status::EvalFailure);
      continue;
    }

    // SUCCESS!!!
    // Output results at log interval
    auto topology =  engine.GetTopology();
    auto stats = topology.GetStats();
    EvaluationResult result = { true, mapping, stats, layout_ };  // Include layout_ in result

    if(log_all_mappings_)
    {
        mutex_->lock(); // Print performance and log the optimal mappings
        topology.PrintOrojenesis(&workload_, orojenesis_csv_file_, mapping, log_mappings_yaml_, log_mappings_verbose_, orojenesis_prefix_, thread_id_);
        mutex_->unlock();
    }
    // Log the equally optimal mappings stats from the previous index factor and clear the index_factor_best_vec
    // Need to have one valid mapping in order to get the SumStats run
    else if (log_orojenesis_mappings_ && total_mappings != 0 && stats_.index_factor_best.valid && SumStats(stats_.index_factor_best.stats.tile_sizes[0]) != SumStats(stats.tile_sizes[0]))
    {
      for (auto &index_factor_best : index_factor_best_vec)
      {

        // Re-evaluate the mapping
        // if (layout_initialized_){
        //   engine.Evaluate(index_factor_best.mapping, workload_, layout_, sparse_optimizations_, crypto_, !diagnostics_on_);
        // }else
        engine.Evaluate(index_factor_best.mapping, workload_, layout_, sparse_optimizations_, crypto_, !diagnostics_on_);

        auto topology = engine.GetTopology();

        mutex_->lock();

        // Print performance and log the optimal mappings
        topology.PrintOrojenesis(&workload_, orojenesis_csv_file_, stats_.index_factor_best.mapping, log_mappings_yaml_, log_mappings_verbose_, orojenesis_prefix_, thread_id_);
        mutex_->unlock();

        // Only print one valid mapping stat if the tiling size is 0 in the inner level
        if (SumStats(stats_.index_factor_best.stats.tile_sizes[0]) == 0)
          break;
      }

      // Reset the best for next permutation/bypassing
      stats_.index_factor_best.valid = false;
      index_factor_best_vec.clear();
    }

    valid_mappings++;
    if (log_stats_)
    {
      mutex_->lock();
      log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                  << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
      mutex_->unlock();
    }
    invalid_mappings_mapcnstr = 0;
    invalid_mappings_eval = 0;
    search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

    bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
    if (log_suboptimal_ && total_mappings != 0 && log_interval_ > 0 && total_mappings % log_interval_ == 0)
    {
      mutex_->lock();
      if (is_sparse_topology)
      {
        log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                  << " Utilization = " << std::setw(4) << OUT_PERCENT(stats.utilization)
                  << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                  << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                  << " | " << mapping.PrintCompact()
                  << " | Cycles = " << stats.cycles
                  << std::endl;
      }
      else
      {
        log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                  << " Utilization = " << std::setw(4) << OUT_PERCENT(stats.utilization)
                  << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                  << " | " << mapping.PrintCompact()
                  << " | Cycles = " << stats.cycles
                  << std::endl;
      }
      mutex_->unlock();
    }

    // Update index factor best
    if (log_orojenesis_mappings_)
    {
      if (stats_.index_factor_best.UpdateIfBetter(result, optimization_metrics_))
      {
        index_factor_best_vec.clear();
        index_factor_best_vec.push_back(stats_.index_factor_best);
      }
      else if (stats_.index_factor_best.UpdateIfEqual(result, optimization_metrics_))
      {
        index_factor_best_vec.push_back(stats_.index_factor_best);
      }
    }

    // Is the new mapping "better" than the previous best mapping?
    if (stats_.thread_best.UpdateIfBetter(result, optimization_metrics_))
    {
      if (log_stats_)
      {
        // FIXME: improvement only captures the primary stat.
        double improvement = stats_.thread_best.valid ?
          (Cost(stats_.thread_best.stats, optimization_metrics_.at(0)) - Cost(stats, optimization_metrics_.at(0))) /
          Cost(stats_.thread_best.stats, optimization_metrics_.at(0)) : 1.0;
        mutex_->lock();
        log_stream_ << "[" << thread_id_ << "] UPDATE " << total_mappings << " " << valid_mappings
                    << " " << mappings_since_last_best_update << " " << improvement << std::endl;
        mutex_->unlock();
      }

      if (!log_suboptimal_)
      {
        mutex_->lock();
        if (is_sparse_topology)
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                    << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                    << " | pJ/Algorithmic-Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                    << " | pJ/Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                    << " | " << mapping.PrintCompact()
                    << " | Cycles = " << stats.cycles
                    << std::endl;
        }
        else
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                    << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                    << " | pJ/Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                    << " | " << mapping.PrintCompact()
                    << " | Cycles = " << stats.cycles
                    << std::endl;
        }        mutex_->unlock();
      }

      mappings_since_last_best_update = 0;
    }
    else
    {
      // If the only change in this mapping vs. the previous mapping was in
      // its dataspace bypass scheme, then we may not want to make this
      // failure count towards the timeout termination trigger.
      if (penalize_consecutive_bypass_fails_ || !only_bypass_changed)
      {
        mappings_since_last_best_update++;
      }
    }
  } // while ()
}