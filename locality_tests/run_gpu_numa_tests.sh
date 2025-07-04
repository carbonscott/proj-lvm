#!/bin/bash

# GPU NUMA Test Runner Script
# This script helps run GPU NUMA tests with proper binding configurations and validation

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default test parameters
NUM_SAMPLES=1000
TENSOR_SHAPE="3 224 224"
BATCH_SIZE=10
WARMUP_SAMPLES=100
MEMORY_SIZE_MB=512
TIMEOUT_S=600
NSYS_REPORT_DIR="nsys_reports"
COMPILE_MODEL=false

print_header() {
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}================================${NC}"
}

print_test() {
    echo -e "${YELLOW}--- $1 ---${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_info() {
    echo -e "${BLUE}INFO: $1${NC}"
}

validate_numa_node() {
    local numa_node=$1
    local silent=${2:-false}

    # Test if we can bind to this NUMA node
    numactl --cpunodebind=$numa_node --membind=$numa_node true 2>/dev/null
    if [ $? -ne 0 ]; then
        if [ "$silent" != "true" ]; then
            print_error "NUMA node $numa_node is not accessible for binding. Skipping." >&2
        fi
        return 1
    fi
    return 0
}

get_available_numa_nodes() {
    local available_nodes=()

    # Get the maximum NUMA node number
    local max_node=$(numactl --hardware 2>/dev/null | grep "available:" | sed 's/.*: \([0-9]*\) nodes.*/\1/' | head -1)
    if [ -z "$max_node" ]; then
        max_node=4  # fallback
    fi

    # Test each node (silent mode to avoid stdout pollution)
    for (( i=0; i<$max_node; i++ )); do
        if validate_numa_node $i true; then
            available_nodes+=($i)
        fi
    done

    echo "${available_nodes[@]}"
}

get_available_gpus() {
    local available_gpus=()

    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "0" # fallback to GPU 0
        return
    fi

    # Get GPU count
    local gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ -z "$gpu_count" ] || [ "$gpu_count" -eq 0 ]; then
        echo "0" # fallback to GPU 0
        return
    fi

    # List available GPUs
    for (( i=0; i<$gpu_count; i++ )); do
        available_gpus+=($i)
    done

    echo "${available_gpus[@]}"
}

get_gpu_numa_affinity() {
    local gpu_id=$1

    # Try to get GPU NUMA affinity from nvidia-smi topo
    local numa_affinity=$(nvidia-smi topo -m 2>/dev/null | grep "^GPU${gpu_id}" | awk '{print $(NF-1)}' | head -1)

    # If we couldn't parse it or it's N/A, return unknown
    if [ -z "$numa_affinity" ] || [ "$numa_affinity" = "N/A" ] || [ "$numa_affinity" = "GPU" ]; then
        echo "unknown"
    else
        echo "$numa_affinity"
    fi
}

check_dependencies() {
    print_header "Checking Dependencies"

    # Check if numactl is available
    if ! command -v numactl &> /dev/null; then
        print_error "numactl is not installed. Please install it first."
        echo "On Ubuntu/Debian: sudo apt-get install numactl"
        echo "On RHEL/CentOS: sudo yum install numactl"
        exit 1
    fi

    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi is not available. Please install NVIDIA drivers."
        exit 1
    fi

    # Check if nsys is available
    if ! command -v nsys &> /dev/null; then
        print_error "nsys is not available. Please install NVIDIA Nsight Systems."
        echo "Download from: https://developer.nvidia.com/nsight-systems"
        exit 1
    fi

    # Check if Python packages are available
    python3 -c "import torch, psutil, numpy" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "Required Python packages not found."
        echo "Please install: pip install torch psutil numpy"
        exit 1
    fi

    # Check CUDA availability
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "CUDA is not available in PyTorch."
        exit 1
    fi

    # Check for vit-pytorch (optional - only needed for depth > 0)
    python3 -c "import vit_pytorch" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_info "vit-pytorch not found. No-op mode (depth=0) will work, but ViT inference requires:"
        echo "  pip install vit-pytorch"
    fi

    # Check for torch.compile availability if compilation is enabled
    if [ "$COMPILE_MODEL" = true ]; then
        python3 -c "import torch; assert hasattr(torch, 'compile'), 'torch.compile not available'" 2>/dev/null
        if [ $? -ne 0 ]; then
            print_error "torch.compile is not available. Requires PyTorch 2.0+."
            echo "Either upgrade PyTorch or run without --compile-model"
            exit 1
        fi
        print_info "torch.compile is available - will use model compilation"
    fi

    echo "All dependencies satisfied."
}

show_gpu_numa_topology() {
    print_header "GPU and NUMA Topology"

    echo "NUMA Topology:"
    numactl --hardware
    echo ""

    echo "GPU Topology:"
    nvidia-smi topo -m
    echo ""

    echo "GPU-NUMA Mapping:"
    available_gpus=($(get_available_gpus))
    for gpu in "${available_gpus[@]}"; do
        numa_affinity=$(get_gpu_numa_affinity $gpu)
        echo "  GPU $gpu -> NUMA $numa_affinity"
    done
}

run_gpu_pipeline_test() {
    local gpu_id=$1
    local numa_node=$2
    local patch_size=$3
    local depth=$4
    local heads=$5
    local dim=$6
    local mlp_dim=$7

    local compile_flag=""
    local compile_suffix=""
    if [ "$COMPILE_MODEL" = true ]; then
        compile_flag="--compile-model"
        compile_suffix=" [COMPILED]"
    fi

    if [ $depth -eq 0 ]; then
        local test_name="GPU Pipeline Test: GPU${gpu_id} on NUMA${numa_node} (No-op: H2D/D2H only)${compile_suffix}"
    else
        local test_name="GPU Pipeline Test: GPU${gpu_id} on NUMA${numa_node} (ViT: ${patch_size}/${depth}/${heads}/${dim}/${mlp_dim})${compile_suffix}"
    fi

    print_test "$test_name"

    # Validate NUMA node
    if ! validate_numa_node $numa_node; then
        echo "Skipping test due to NUMA node $numa_node not being accessible"
        return 1
    fi

    # Clean up any existing processes
    pkill -f "gpu_numa_pipeline_test" 2>/dev/null
    sleep 1

    echo "Starting pipeline test on GPU $gpu_id with NUMA node $numa_node..."

    # Create nsys report directory if it doesn't exist
    mkdir -p "$NSYS_REPORT_DIR"

    if [ $depth -eq 0 ]; then
        local nsys_output="${NSYS_REPORT_DIR}/h2d_d2h_gpu${gpu_id}_numa${numa_node}"
    else
        local nsys_output="${NSYS_REPORT_DIR}/pipeline_gpu${gpu_id}_numa${numa_node}_vit${patch_size}x${depth}x${dim}"
    fi

    # Add compilation suffix to report name if enabled
    if [ "$COMPILE_MODEL" = true ]; then
        nsys_output="${nsys_output}_compiled"
    fi

    timeout $TIMEOUT_S nsys profile -o "$nsys_output" --trace=cuda,nvtx \
        numactl --cpunodebind=$numa_node --membind=$numa_node \
        python3 gpu_numa_pipeline_test.py \
        --gpu-id $gpu_id \
        --shape $TENSOR_SHAPE \
        --num-samples $NUM_SAMPLES \
        --batch-size $BATCH_SIZE \
        --warmup-samples $WARMUP_SAMPLES \
        --memory-size-mb $MEMORY_SIZE_MB \
        --vit-patch-size $patch_size \
        --vit-depth $depth \
        --vit-heads $heads \
        --vit-dim $dim \
        --vit-mlp-dim $mlp_dim \
        $compile_flag

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}$test_name completed successfully${NC}"
        echo "NSYS report saved: ${nsys_output}.nsys-rep"
    else
        print_error "$test_name failed with exit code $exit_code"
        return 1
    fi
    echo ""
}

run_all_pipeline_tests() {
    print_header "Running GPU Pipeline Tests"

    # Get available NUMA nodes and GPUs
    available_nodes=($(get_available_numa_nodes))
    available_gpus=($(get_available_gpus))

    echo "Available NUMA nodes: ${available_nodes[@]}"
    echo "Available GPUs: ${available_gpus[@]}"
    echo "Model compilation: $COMPILE_MODEL"
    echo ""

    if [ ${#available_nodes[@]} -lt 1 ]; then
        print_error "Need at least 1 NUMA node for testing. Found: ${#available_nodes[@]}"
        return 1
    fi

    if [ ${#available_gpus[@]} -lt 1 ]; then
        print_error "Need at least 1 GPU for testing. Found: ${#available_gpus[@]}"
        return 1
    fi

    # Different ViT configurations to test varying compute loads
    # Format: patch_size depth heads dim mlp_dim
    declare -a vit_configs=(
        "32 0 8 256 1024"     # No-op: H2D/D2H only
        "32 4 8 256 1024"     # Light: Small model, fast compute
        "32 6 8 512 2048"     # Medium: Balanced model
        "16 8 12 768 3072"    # Heavy: Larger model, more compute
        "16 12 16 1024 4096"  # Very Heavy: Large model for compute-bound tests
    )

    # Test each GPU with each NUMA node and ViT configuration
    for gpu in "${available_gpus[@]}"; do
        gpu_numa_affinity=$(get_gpu_numa_affinity $gpu)
        print_info "GPU $gpu has NUMA affinity: $gpu_numa_affinity"

        for numa_node in "${available_nodes[@]}"; do
            for vit_config in "${vit_configs[@]}"; do
                # Parse ViT config
                read -ra VIT_PARAMS <<< "$vit_config"
                local patch_size=${VIT_PARAMS[0]}
                local depth=${VIT_PARAMS[1]}
                local heads=${VIT_PARAMS[2]}
                local dim=${VIT_PARAMS[3]}
                local mlp_dim=${VIT_PARAMS[4]}

                run_gpu_pipeline_test $gpu $numa_node $patch_size $depth $heads $dim $mlp_dim
            done
        done
    done
}

run_focused_tests() {
    local gpu_id=$1
    local numa_nodes=${2:-"all"}

    print_header "Running Focused Tests for GPU $gpu_id"

    # Get available NUMA nodes
    available_nodes=($(get_available_numa_nodes))

    if [ "$numa_nodes" = "all" ]; then
        test_nodes=("${available_nodes[@]}")
    else
        # Parse comma-separated list
        IFS=',' read -ra test_nodes <<< "$numa_nodes"
    fi

    echo "Testing GPU $gpu_id with NUMA nodes: ${test_nodes[@]}"
    echo "Model compilation: $COMPILE_MODEL"
    echo ""

    # Run pipeline tests with focused ViT configurations
    declare -a focused_vit_configs=(
        "32 0 8 256 1024"     # No-op: H2D/D2H only
        "32 6 8 512 2048"     # Medium: Balanced
        "16 8 12 768 3072"    # Heavy: More compute-intensive
    )

    for numa_node in "${test_nodes[@]}"; do
        for vit_config in "${focused_vit_configs[@]}"; do
            # Parse ViT config
            read -ra VIT_PARAMS <<< "$vit_config"
            local patch_size=${VIT_PARAMS[0]}
            local depth=${VIT_PARAMS[1]}
            local heads=${VIT_PARAMS[2]}
            local dim=${VIT_PARAMS[3]}
            local mlp_dim=${VIT_PARAMS[4]}

            run_gpu_pipeline_test $gpu_id $numa_node $patch_size $depth $heads $dim $mlp_dim
        done
    done
}

validate_gpu_numa_setup() {
    print_header "GPU NUMA Setup Validation"

    # Get available NUMA nodes and GPUs
    available_nodes=($(get_available_numa_nodes))
    available_gpus=($(get_available_gpus))

    echo "Testing NUMA node accessibility..."
    echo ""

    local max_node=$(numactl --hardware 2>/dev/null | grep "available:" | sed 's/.*: \([0-9]*\) nodes.*/\1/' | head -1)
    if [ -z "$max_node" ]; then
        max_node=4  # fallback
    fi

    for (( i=0; i<$max_node; i++ )); do
        echo -n "NUMA node $i: "
        if validate_numa_node $i; then
            echo -e "${GREEN}✓ Available${NC}"
        else
            echo -e "${RED}✗ Not accessible${NC}"
        fi
    done

    echo ""
    echo "Available NUMA nodes: ${available_nodes[@]}"
    echo "Available GPUs: ${available_gpus[@]}"
    echo ""

    echo "GPU-NUMA Affinity Analysis:"
    for gpu in "${available_gpus[@]}"; do
        gpu_numa_affinity=$(get_gpu_numa_affinity $gpu)
        echo -n "GPU $gpu (NUMA $gpu_numa_affinity): "

        if [ "$gpu_numa_affinity" != "unknown" ] && validate_numa_node $gpu_numa_affinity true; then
            echo -e "${GREEN}✓ Optimal NUMA node accessible${NC}"
        elif [ "$gpu_numa_affinity" = "unknown" ]; then
            echo -e "${YELLOW}? NUMA affinity unknown${NC}"
        else
            echo -e "${RED}✗ Optimal NUMA node not accessible${NC}"
        fi
    done

    echo ""
    echo "Recommended test configurations:"
    if [ ${#available_nodes[@]} -ge 1 ] && [ ${#available_gpus[@]} -ge 1 ]; then
        echo "- Pipeline tests: Ready (${#available_gpus[@]} GPUs × ${#available_nodes[@]} NUMA nodes)"
        echo "  * No-op mode (depth=0): Pure H2D/D2H performance"
        echo "  * ViT modes (depth>0): Compute + memory transfer pipeline"
        if [ "$COMPILE_MODEL" = true ]; then
            echo "  * Model compilation: ENABLED (reduced kernel overhead)"
        else
            echo "  * Model compilation: DISABLED (add --compile-model to enable)"
        fi

        # Suggest interesting test cases
        echo ""
        echo "Interesting test cases to focus on:"
        for gpu in "${available_gpus[@]}"; do
            gpu_numa_affinity=$(get_gpu_numa_affinity $gpu)
            if [ "$gpu_numa_affinity" != "unknown" ]; then
                echo "  GPU $gpu: Compare NUMA $gpu_numa_affinity (local) vs other NUMA nodes (remote)"
            fi
        done
    else
        echo "- Tests cannot run: insufficient GPUs or NUMA nodes"
    fi
}

usage() {
    echo "Usage: $0 [OPTIONS] [TEST_TYPE]"
    echo ""
    echo "TEST_TYPE:"
    echo "  pipeline      Run pipeline tests with ViT or no-op compute (default)"
    echo "  topology      Show GPU and NUMA topology only"
    echo "  validate      Validate GPU-NUMA setup only"
    echo "  focused GPU_ID [NUMA_NODES]  Run focused tests for specific GPU"
    echo ""
    echo "OPTIONS:"
    echo "  -n SAMPLES          Number of samples per test (default: $NUM_SAMPLES)"
    echo "  -s \"C H W\"          Tensor shape (default: \"$TENSOR_SHAPE\")"
    echo "  -b BATCH_SIZE       Batch size (default: $BATCH_SIZE)"
    echo "  -w WARMUP           Warmup samples (default: $WARMUP_SAMPLES)"
    echo "  -m MEMORY_MB        Memory pool size in MB (default: $MEMORY_SIZE_MB)"
    echo "  -r NSYS_DIR         NSYS reports directory (default: $NSYS_REPORT_DIR)"
    echo "  -t TIMEOUT_S        Test timeout in seconds (default: $TIMEOUT_S)"
    echo "  -c, --compile-model Enable torch.compile for reduced kernel overhead"
    echo "  -h                  Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                              # Run all pipeline tests (including no-op)"
    echo "  $0 pipeline                     # Run pipeline tests"
    echo "  $0 --compile-model pipeline     # Run pipeline tests with model compilation"
    echo "  $0 validate                     # Check GPU-NUMA setup"
    echo "  $0 focused 3                    # Run focused tests for GPU 3"
    echo "  $0 focused 3 0,2                # Test GPU 3 with NUMA nodes 0,2"
    echo "  $0 -n 2000 -c pipeline         # Custom parameters with compilation"
    echo "  $0 -r my_reports pipeline       # Custom nsys output directory"
    echo ""
    echo "ViT Pipeline Configurations:"
    echo "  No-op compute:  patch_size=32, depth=0           (H2D/D2H only)"
    echo "  Light compute:  patch_size=32, depth=4, dim=256  (memory transfer dominates)"
    echo "  Medium compute: patch_size=32, depth=6, dim=512  (balanced)"
    echo "  Heavy compute:  patch_size=16, depth=8, dim=768  (compute dominates)"
    echo "  Very Heavy:     patch_size=16, depth=12, dim=1024 (compute-bound)"
    echo ""
    echo "Model Compilation (torch.compile):"
    echo "  - Reduces CUDA kernel overhead, especially beneficial for small ViTs"
    echo "  - Requires PyTorch 2.0+ and may increase first-run compilation time"
    echo "  - Most effective for depth=4-8 models where kernel overhead is significant"
}

# Parse command line arguments - FIXED VERSION
while getopts "n:s:b:w:m:r:t:ch" opt; do
    case $opt in
        n)
            NUM_SAMPLES=$OPTARG
            ;;
        s)
            TENSOR_SHAPE=$OPTARG
            ;;
        b)
            BATCH_SIZE=$OPTARG
            ;;
        w)
            WARMUP_SAMPLES=$OPTARG
            ;;
        m)
            MEMORY_SIZE_MB=$OPTARG
            ;;
        r)
            NSYS_REPORT_DIR=$OPTARG
            ;;
        t)
            TIMEOUT_S=$OPTARG
            ;;
        c)
            COMPILE_MODEL=true
            ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            exit 1
            ;;
    esac
done

shift $((OPTIND-1))

# Handle long options manually (since getopts doesn't support them)
while [[ $# -gt 0 ]]; do
    case $1 in
        --compile-model)
            COMPILE_MODEL=true
            shift
            ;;
        -*)
            echo "Unknown option $1"
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

TEST_TYPE=${1:-pipeline}

# Main execution
check_dependencies

case $TEST_TYPE in
    topology)
        show_gpu_numa_topology
        ;;
    validate)
        show_gpu_numa_topology
        echo ""
        validate_gpu_numa_setup
        ;;
    pipeline)
        show_gpu_numa_topology
        run_all_pipeline_tests
        ;;
    focused)
        if [ $# -lt 2 ]; then
            print_error "GPU ID required for focused tests"
            usage
            exit 1
        fi
        GPU_ID=$2
        NUMA_NODES=${3:-"all"}
        show_gpu_numa_topology
        run_focused_tests $GPU_ID $NUMA_NODES
        ;;
    *)
        print_error "Unknown test type: $TEST_TYPE"
        usage
        exit 1
        ;;
esac

print_header "Tests Completed"
echo "Review the results above to understand GPU-NUMA effects on your system."
echo ""
echo "NSYS reports saved in: $NSYS_REPORT_DIR/"
echo "For detailed analysis, use nsys to profile the generated .nsys-rep files:"
echo "  nsys stats $NSYS_REPORT_DIR/*.nsys-rep"
echo "  nsys-ui $NSYS_REPORT_DIR/*.nsys-rep  # For GUI analysis"
if [ "$COMPILE_MODEL" = true ]; then
    echo ""
    echo "Model compilation was ENABLED. Reports include '_compiled' suffix."
    echo "Compare with non-compiled runs to measure kernel overhead reduction."
fi
