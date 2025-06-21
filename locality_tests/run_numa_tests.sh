#!/bin/bash

# NUMA Test Runner Script
# This script helps run the NUMA tests with proper binding configurations and validation

# Configuration based on your topology
NIC_NUMA_NODE=2  # enp69s0f0 is on NUMA node 2

# Default test parameters
NUM_SAMPLES=1000
NETWORK_PORT=5555
IPC_PATH="/tmp/numa_ipc_test"
TENSOR_SHAPE="1 224 224"
BATCH_BUFFER_SIZE=10
MEMORY_SIZE_MB=100
TIMEOUT_MS=10000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

validate_numa_node() {
    local numa_node=$1

    # Test if we can bind to this NUMA node
    numactl --cpunodebind=$numa_node --membind=$numa_node true 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "NUMA node $numa_node is not accessible for binding. Skipping."
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

    # Test each node
    for (( i=0; i<$max_node; i++ )); do
        if validate_numa_node $i; then
            available_nodes+=($i)
        fi
    done

    echo "${available_nodes[@]}"
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

    # Check if Python packages are available
    python3 -c "import torch, pynng, psutil, numpy" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "Required Python packages not found."
        echo "Please install: pip install torch pynng psutil numpy"
        exit 1
    fi

    echo "All dependencies satisfied."
}

show_numa_topology() {
    print_header "NUMA Topology"
    numactl --hardware
    echo ""
    echo "CPU info:"
    lscpu | grep -E "NUMA|Socket|Core|Thread"
}

run_network_test() {
    local pusher_numa=$1
    local puller_numa=$2
    local test_name="Network Test: Pusher NUMA${pusher_numa} -> Puller NUMA${puller_numa}"

    print_test "$test_name"

    # Validate NUMA nodes before starting
    if ! validate_numa_node $pusher_numa; then
        echo "Skipping test due to pusher NUMA node $pusher_numa not being accessible"
        return 1
    fi

    if ! validate_numa_node $puller_numa; then
        echo "Skipping test due to puller NUMA node $puller_numa not being accessible"
        return 1
    fi

    # Clean up any existing processes
    pkill -f "numa_network_test" 2>/dev/null
    sleep 1

    # Start pusher in background with error checking
    echo "Starting pusher on NUMA node $pusher_numa..."
    numactl --cpunodebind=$pusher_numa --membind=$pusher_numa \
        python3 numa_network_test_pusher.py \
        --address "tcp://0.0.0.0:$NETWORK_PORT" \
        --num-samples $NUM_SAMPLES \
        --shape $TENSOR_SHAPE \
        --batch-buffer-size $BATCH_BUFFER_SIZE \
        --memory-size-mb $MEMORY_SIZE_MB \
        --nic-numa $NIC_NUMA_NODE &
    PUSHER_PID=$!

    # Check if pusher started successfully
    sleep 2
    if ! kill -0 $PUSHER_PID 2>/dev/null; then
        print_error "Pusher failed to start"
        return 1
    fi

    # Wait a bit more for network socket to be ready
    sleep 3

    # Start puller with timeout
    echo "Starting puller on NUMA node $puller_numa..."
    timeout 60 numactl --cpunodebind=$puller_numa --membind=$puller_numa \
        python3 numa_network_test_puller.py \
        --address "tcp://127.0.0.1:$NETWORK_PORT" \
        --expected-samples $NUM_SAMPLES \
        --timeout $TIMEOUT_MS \
        --batch-buffer-size $BATCH_BUFFER_SIZE \
        --memory-size-mb $MEMORY_SIZE_MB \
        --nic-numa $NIC_NUMA_NODE

    local puller_exit_code=$?

    # Clean up pusher
    kill $PUSHER_PID 2>/dev/null
    wait $PUSHER_PID 2>/dev/null

    if [ $puller_exit_code -eq 0 ]; then
        echo -e "${GREEN}$test_name completed successfully${NC}"
    else
        print_error "$test_name failed with exit code $puller_exit_code"
        return 1
    fi
    echo ""
}

run_ipc_test() {
    local pusher_numa=$1
    local puller_numa=$2
    local test_name="IPC Test: Pusher NUMA${pusher_numa} -> Puller NUMA${puller_numa}"

    print_test "$test_name"

    # Validate NUMA nodes before starting
    if ! validate_numa_node $pusher_numa; then
        echo "Skipping test due to pusher NUMA node $pusher_numa not being accessible"
        return 1
    fi

    if ! validate_numa_node $puller_numa; then
        echo "Skipping test due to puller NUMA node $puller_numa not being accessible"
        return 1
    fi

    # Clean up any existing processes and IPC files
    pkill -f "numa_ipc_test" 2>/dev/null
    rm -f $IPC_PATH 2>/dev/null
    sleep 1

    # Start pusher in background with error checking
    echo "Starting pusher on NUMA node $pusher_numa..."
    numactl --cpunodebind=$pusher_numa --membind=$pusher_numa \
        python3 numa_ipc_test_pusher.py \
        --ipc-path $IPC_PATH \
        --num-samples $NUM_SAMPLES \
        --shape $TENSOR_SHAPE \
        --batch-buffer-size $BATCH_BUFFER_SIZE \
        --memory-size-mb $MEMORY_SIZE_MB &
    PUSHER_PID=$!

    # Check if pusher started successfully
    sleep 2
    if ! kill -0 $PUSHER_PID 2>/dev/null; then
        print_error "Pusher failed to start"
        return 1
    fi

    # Wait a bit more for IPC socket to be ready
    sleep 3

    # Start puller with timeout for connection
    echo "Starting puller on NUMA node $puller_numa..."
    timeout 60 numactl --cpunodebind=$puller_numa --membind=$puller_numa \
        python3 numa_ipc_test_puller.py \
        --ipc-path $IPC_PATH \
        --expected-samples $NUM_SAMPLES \
        --timeout $TIMEOUT_MS \
        --batch-buffer-size $BATCH_BUFFER_SIZE \
        --memory-size-mb $MEMORY_SIZE_MB

    local puller_exit_code=$?

    # Clean up
    kill $PUSHER_PID 2>/dev/null
    wait $PUSHER_PID 2>/dev/null
    rm -f $IPC_PATH 2>/dev/null

    if [ $puller_exit_code -eq 0 ]; then
        echo -e "${GREEN}$test_name completed successfully${NC}"
    else
        print_error "$test_name failed with exit code $puller_exit_code"
        return 1
    fi
    echo ""
}

run_all_network_tests() {
    print_header "Running Network Socket Tests"

    # Get available NUMA nodes
    available_nodes=($(get_available_numa_nodes))
    echo "Available NUMA nodes: ${available_nodes[@]}"
    echo "Testing performance with NIC on NUMA node $NIC_NUMA_NODE"

    if [ ${#available_nodes[@]} -lt 1 ]; then
        print_error "Need at least 1 NUMA node for testing. Found: ${#available_nodes[@]}"
        return 1
    fi

    # Check if NIC_NUMA_NODE is accessible
    if ! validate_numa_node $NIC_NUMA_NODE; then
        print_error "NIC NUMA node $NIC_NUMA_NODE is not accessible. Network tests may not be meaningful."
        echo "Consider updating NIC_NUMA_NODE variable in the script."
    fi

    echo ""

    # Test 1: Same NUMA node as NIC (if NIC node is available)
    if validate_numa_node $NIC_NUMA_NODE; then
        run_network_test $NIC_NUMA_NODE $NIC_NUMA_NODE
    fi

    # Test 2: Cross-NUMA tests involving NIC node
    for node in "${available_nodes[@]}"; do
        if [ $node -ne $NIC_NUMA_NODE ]; then
            run_network_test $node $NIC_NUMA_NODE
            run_network_test $NIC_NUMA_NODE $node
        fi
    done

    # Test 3: Cross-NUMA tests not involving NIC node (if we have enough nodes)
    if [ ${#available_nodes[@]} -ge 3 ]; then
        for (( i=0; i<${#available_nodes[@]}; i++ )); do
            for (( j=i+1; j<${#available_nodes[@]}; j++ )); do
                local node1=${available_nodes[$i]}
                local node2=${available_nodes[$j]}
                # Skip if either is the NIC node (already tested above)
                if [ $node1 -ne $NIC_NUMA_NODE ] && [ $node2 -ne $NIC_NUMA_NODE ]; then
                    run_network_test $node1 $node2
                fi
            done
        done
    fi
}

run_all_ipc_tests() {
    print_header "Running IPC Socket Tests"
    echo "Testing cross-NUMA memory access performance"

    # Get available NUMA nodes
    available_nodes=($(get_available_numa_nodes))
    echo "Available NUMA nodes: ${available_nodes[@]}"

    if [ ${#available_nodes[@]} -lt 2 ]; then
        print_error "Need at least 2 NUMA nodes for testing. Found: ${#available_nodes[@]}"
        return 1
    fi

    echo ""

    # Test 1: Same NUMA node (for each available node)
    for node in "${available_nodes[@]}"; do
        run_ipc_test $node $node
    done

    # Test 2: Cross-NUMA tests (if we have enough nodes)
    if [ ${#available_nodes[@]} -ge 2 ]; then
        for (( i=0; i<${#available_nodes[@]}; i++ )); do
            for (( j=i+1; j<${#available_nodes[@]}; j++ )); do
                run_ipc_test ${available_nodes[$i]} ${available_nodes[$j]}
                run_ipc_test ${available_nodes[$j]} ${available_nodes[$i]}
            done
        done
    fi
}

usage() {
    echo "Usage: $0 [OPTIONS] [TEST_TYPE]"
    echo ""
    echo "TEST_TYPE:"
    echo "  network    Run network socket tests"
    echo "  ipc        Run IPC socket tests"
    echo "  all        Run all tests (default)"
    echo "  topology   Show NUMA topology only"
    echo "  validate   Validate NUMA node accessibility only"
    echo ""
    echo "OPTIONS:"
    echo "  -n SAMPLES         Number of samples per test (default: $NUM_SAMPLES)"
    echo "  -p PORT            Network port (default: $NETWORK_PORT)"
    echo "  -i PATH            IPC socket path (default: $IPC_PATH)"
    echo "  -s \"C H W\"         Tensor shape (default: \"$TENSOR_SHAPE\")"
    echo "  -b BUFFER_SIZE     Batch-buffer size (default: $BATCH_BUFFER_SIZE)"
    echo "  -m MEMORY_MB       Memory pool size in MB (default: $MEMORY_SIZE_MB)"
    echo "  -t TIMEOUT_MS      Socket timeout in ms (default: $TIMEOUT_MS)"
    echo "  -h                 Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all tests"
    echo "  $0 network                            # Run network tests only"
    echo "  $0 ipc                                # Run IPC tests only"
    echo "  $0 validate                           # Check which NUMA nodes are accessible"
    echo "  $0 -n 2000 network                   # Run network tests with 2000 samples"
    echo "  $0 -s \"3 512 512\" -b 20 all         # Custom tensor shape and buffer size"
}

validate_numa_setup() {
    print_header "NUMA Node Validation"

    # Get available NUMA nodes
    available_nodes=($(get_available_numa_nodes))
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
    echo "NIC NUMA node (configured): $NIC_NUMA_NODE"

    if validate_numa_node $NIC_NUMA_NODE; then
        echo -e "NIC NUMA node accessibility: ${GREEN}✓ Available${NC}"
    else
        echo -e "NIC NUMA node accessibility: ${RED}✗ Not accessible${NC}"
        echo -e "${YELLOW}WARNING: Network tests may not be meaningful${NC}"
    fi

    echo ""
    echo "Recommended test configurations:"
    if [ ${#available_nodes[@]} -ge 2 ]; then
        echo "- IPC tests: Ready (${#available_nodes[@]} nodes available)"
        echo "- Network tests: Ready"
    elif [ ${#available_nodes[@]} -eq 1 ]; then
        echo "- IPC tests: Limited (only same-node tests possible)"
        echo "- Network tests: Limited (no cross-NUMA comparison)"
    else
        echo "- No NUMA nodes accessible - tests will fail"
    fi
}

# Parse command line arguments
while getopts "n:p:i:s:b:m:t:h" opt; do
    case $opt in
        n)
            NUM_SAMPLES=$OPTARG
            ;;
        p)
            NETWORK_PORT=$OPTARG
            ;;
        i)
            IPC_PATH=$OPTARG
            ;;
        s)
            TENSOR_SHAPE=$OPTARG
            ;;
        b)
            BATCH_BUFFER_SIZE=$OPTARG
            ;;
        m)
            MEMORY_SIZE_MB=$OPTARG
            ;;
        t)
            TIMEOUT_MS=$OPTARG
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
TEST_TYPE=${1:-all}

# Main execution
check_dependencies

case $TEST_TYPE in
    topology)
        show_numa_topology
        ;;
    validate)
        show_numa_topology
        echo ""
        validate_numa_setup
        ;;
    network)
        show_numa_topology
        run_all_network_tests
        ;;
    ipc)
        show_numa_topology
        run_all_ipc_tests
        ;;
    all)
        show_numa_topology
        run_all_network_tests
        run_all_ipc_tests
        ;;
    *)
        print_error "Unknown test type: $TEST_TYPE"
        usage
        exit 1
        ;;
esac

print_header "Tests Completed"
echo "Review the results above to understand NUMA effects on your system."
