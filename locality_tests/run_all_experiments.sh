#!/bin/bash
# Comprehensive ViT Inference Experiment Runner with Corrected Timing
# Author: Claude Code Assistant
# Date: July 14, 2025
# 
# This script runs all research questions (RQ1-RQ5) with proper error handling,
# process management, and automatic analysis pipeline execution.

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
RESULTS_DIR="${SCRIPT_DIR}/experiments"
ANALYSIS_DIR="${SCRIPT_DIR}/analysis"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Research Questions Configuration
declare -A RQ_CONFIGS=(
    ["rq1"]="rq1_model_architecture_impact"
    ["rq2"]="rq2_resolution_scaling"
    ["rq3"]="rq3_compilation_optimization"
    ["rq4"]="rq4_numa_locality_effects"
    ["rq5"]="rq5_pipeline_design"
)

declare -A RQ_DESCRIPTIONS=(
    ["rq1"]="Model Architecture Impact on Inference Performance"
    ["rq2"]="Input Resolution Scaling Characteristics"
    ["rq3"]="Compilation Optimization Effectiveness"
    ["rq4"]="NUMA Binding and Memory Locality Effects"
    ["rq5"]="Asynchronous Pipeline Optimization Strategies"
)

# Default: run all research questions
DEFAULT_RQS="rq1 rq2 rq3 rq4 rq5"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [RQ1] [RQ2] [RQ3] [RQ4] [RQ5]

Run comprehensive ViT inference experiments with corrected timing methodology.

OPTIONS:
    --clean         Kill any existing experiment processes before starting
    --no-analysis   Skip automatic analysis after experiments complete
    --help          Show this help message

RESEARCH QUESTIONS:
    rq1             Model Architecture Impact (39 configs)
    rq2             Resolution Scaling (17 configs)  
    rq3             Compilation Optimization (26 configs)
    rq4             NUMA Locality Effects (31 configs)
    rq5             Pipeline Design (20 configs)

EXAMPLES:
    $0                      # Run all research questions
    $0 rq1 rq2             # Run only RQ1 and RQ2
    $0 --clean rq3         # Clean existing processes, then run RQ3
    $0 --no-analysis       # Run all but skip analysis

OUTPUTS:
    logs/           Individual experiment logs (rq1_run.log, etc.)
    experiments/    Raw experimental results and nsys reports
    analysis/       Processed analysis results and reports

EOF
}

cleanup_processes() {
    log_info "Cleaning up existing experiment processes..."
    
    # Kill hydra experiment runners
    if pgrep -f "hydra_experiment_runner.py" > /dev/null; then
        log_warning "Killing existing hydra experiment processes..."
        pkill -f "hydra_experiment_runner.py" || true
        sleep 5
    fi
    
    # Kill any lingering Python processes
    if pgrep -f "gpu_numa_pipeline_test.py" > /dev/null; then
        log_warning "Killing existing pipeline test processes..."
        pkill -f "gpu_numa_pipeline_test.py" || true
        sleep 3
    fi
    
    log_success "Process cleanup completed"
}

validate_environment() {
    log_info "Validating experiment environment..."
    
    # Check required files
    local required_files=(
        "hydra_experiment_runner.py"
        "gpu_numa_pipeline_test.py"
        "conf/config.yaml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    # Check GPU availability
    if ! nvidia-smi > /dev/null 2>&1; then
        log_error "NVIDIA GPU not available or nvidia-smi not found"
        exit 1
    fi
    
    # Check Python environment
    if ! python -c "import torch; import hydra" > /dev/null 2>&1; then
        log_error "Required Python packages not available (torch, hydra)"
        exit 1
    fi
    
    log_success "Environment validation passed"
}

run_research_question() {
    local rq="$1"
    local config="${RQ_CONFIGS[$rq]}"
    local description="${RQ_DESCRIPTIONS[$rq]}"
    local log_file="${LOG_DIR}/${rq}_run.log"
    
    log_info "Starting $rq: $description"
    log_info "Configuration: $config"
    log_info "Log file: $log_file"
    
    # Create log directory if it doesn't exist
    mkdir -p "$LOG_DIR"
    
    # Run the experiment
    local start_time=$(date +%s)
    if python hydra_experiment_runner.py experiment="$config" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "$rq completed successfully in ${duration}s"
        
        # Check for failures in log
        local failed_count=$(grep -c "failed: [1-9]" "$log_file" || echo "0")
        if [[ "$failed_count" -gt 0 ]]; then
            log_warning "$rq completed with $failed_count failed experiments"
        fi
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "$rq failed after ${duration}s - check $log_file"
        return 1
    fi
}

run_analysis_pipeline() {
    log_info "Starting comprehensive analysis pipeline..."
    
    # Create analysis directory
    mkdir -p "$ANALYSIS_DIR"
    
    # Step 1: Parse basic metrics
    log_info "Step 1: Parsing basic performance metrics..."
    if python basic_metrics_parser.py > "${LOG_DIR}/analysis_basic_metrics.log" 2>&1; then
        log_success "Basic metrics parsing completed"
    else
        log_error "Basic metrics parsing failed - check ${LOG_DIR}/analysis_basic_metrics.log"
    fi
    
    # Step 2: Convert nsys reports to SQLite
    log_info "Step 2: Converting nsys reports to SQLite databases..."
    if python batch_nsys_to_sqlite.py > "${LOG_DIR}/analysis_nsys_conversion.log" 2>&1; then
        log_success "Nsys conversion completed"
    else
        log_warning "Nsys conversion failed - check ${LOG_DIR}/analysis_nsys_conversion.log"
    fi
    
    # Step 3: Run CUPTI analysis
    log_info "Step 3: Running comprehensive CUPTI analysis..."
    if find experiments -name "*.sqlite" | head -5 | xargs -I {} python analyze_latency.py --database {} > "${LOG_DIR}/analysis_cupti.log" 2>&1; then
        log_success "CUPTI analysis completed"
    else
        log_warning "CUPTI analysis failed - check ${LOG_DIR}/analysis_cupti.log"
    fi
    
    # Step 4: Generate research question analysis
    log_info "Step 4: Generating research question analysis..."
    if python research_question_analyzer.py > "${LOG_DIR}/analysis_research_questions.log" 2>&1; then
        log_success "Research question analysis completed"
    else
        log_error "Research question analysis failed - check ${LOG_DIR}/analysis_research_questions.log"
    fi
    
    log_success "Analysis pipeline completed - check ${ANALYSIS_DIR}/ for results"
}

# Main execution
main() {
    local clean_processes=false
    local run_analysis=true
    local rqs_to_run=()
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                clean_processes=true
                shift
                ;;
            --no-analysis)
                run_analysis=false
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            rq[1-5])
                rqs_to_run+=("$1")
                shift
                ;;
            *)
                log_error "Unknown argument: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # If no specific RQs specified, run all
    if [[ ${#rqs_to_run[@]} -eq 0 ]]; then
        read -ra rqs_to_run <<< "$DEFAULT_RQS"
    fi
    
    # Start execution
    log_info "=== ViT Inference Experiment Suite with Corrected Timing ==="
    log_info "Started at: $(date)"
    log_info "Research questions to run: ${rqs_to_run[*]}"
    log_info "Clean processes: $clean_processes"
    log_info "Run analysis: $run_analysis"
    
    # Cleanup if requested
    if [[ "$clean_processes" == true ]]; then
        cleanup_processes
    fi
    
    # Validate environment
    validate_environment
    
    # Run experiments
    local failed_rqs=()
    for rq in "${rqs_to_run[@]}"; do
        if [[ -n "${RQ_CONFIGS[$rq]:-}" ]]; then
            if ! run_research_question "$rq"; then
                failed_rqs+=("$rq")
            fi
        else
            log_error "Unknown research question: $rq"
            failed_rqs+=("$rq")
        fi
    done
    
    # Report results
    log_info "=== Experiment Execution Summary ==="
    log_info "Completed at: $(date)"
    log_info "Successful: $((${#rqs_to_run[@]} - ${#failed_rqs[@]}))"
    log_info "Failed: ${#failed_rqs[@]}"
    
    if [[ ${#failed_rqs[@]} -gt 0 ]]; then
        log_warning "Failed research questions: ${failed_rqs[*]}"
    fi
    
    # Run analysis pipeline if requested
    if [[ "$run_analysis" == true ]] && [[ ${#failed_rqs[@]} -lt ${#rqs_to_run[@]} ]]; then
        run_analysis_pipeline
    elif [[ "$run_analysis" == true ]]; then
        log_warning "Skipping analysis due to experiment failures"
    fi
    
    log_success "=== All operations completed ==="
    log_info "Check logs in: $LOG_DIR"
    log_info "Check results in: $RESULTS_DIR"
    if [[ "$run_analysis" == true ]]; then
        log_info "Check analysis in: $ANALYSIS_DIR"
    fi
}

# Run main function with all arguments
main "$@"