#!/bin/bash

# NUMA Test Runner Script
# This script helps run the NUMA tests with proper binding configurations

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

    # Clean up any existing processes
    pkill -f "numa_network_test" 2>/dev/null
    sleep 1

    # Start pusher in background
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

    # Wait a moment for pusher to start
    sleep 3

    # Start puller
    echo "Starting puller on NUMA node $puller_numa..."
    numactl --cpunodebind=$puller_numa --membind=$puller_numa \
        python3 numa_network_test_puller.py \
        --address "tcp://127.0.0.1:$NETWORK_PORT" \
        --expected-samples $NUM_SAMPLES \
        --timeout $TIMEOUT_MS \
        --batch-buffer-size $BATCH_BUFFER_SIZE \
        --memory-size-mb $MEMORY_SIZE_MB \
        --nic-numa $NIC_NUMA_NODE

    # Clean up pusher
    kill $PUSHER_PID 2>/dev/null
    wait $PUSHER_PID 2>/dev/null

    echo -e "${GREEN}$test_name completed${NC}"
    echo ""
}

run_ipc_test() {
    local pusher_numa=$1
    local puller_numa=$2
    local test_name="IPC Test: Pusher NUMA${pusher_numa} -> Puller NUMA${puller_numa}"

    print_test "$test_name"

    # Clean up any existing processes and IPC files
    pkill -f "numa_ipc_test" 2>/dev/null
    rm -f $IPC_PATH 2>/dev/null
    sleep 1

    # Start pusher in background
    echo "Starting pusher on NUMA node $pusher_numa..."
    numactl --cpunodebind=$pusher_numa --membind=$pusher_numa \
        python3 numa_ipc_test_pusher.py \
        --ipc-path $IPC_PATH \
        --num-samples $NUM_SAMPLES \
        --shape $TENSOR_SHAPE \
        --batch-buffer-size $BATCH_BUFFER_SIZE \
        --memory-size-mb $MEMORY_SIZE_MB &
    PUSHER_PID=$!

    # Wait a moment for pusher to start
    sleep 3

    # Start puller
    echo "Starting puller on NUMA node $puller_numa..."
    numactl --cpunodebind=$puller_numa --membind=$puller_numa \
        python3 numa_ipc_test_puller.py \
        --ipc-path $IPC_PATH \
        --expected-samples $NUM_SAMPLES \
        --timeout $TIMEOUT_MS \
        --batch-buffer-size $BATCH_BUFFER_SIZE \
        --memory-size-mb $MEMORY_SIZE_MB

    # Clean up
    kill $PUSHER_PID 2>/dev/null
    wait $PUSHER_PID 2>/dev/null
    rm -f $IPC_PATH 2>/dev/null

    echo -e "${GREEN}$test_name completed${NC}"
    echo ""
}

run_all_network_tests() {
    print_header "Running Network Socket Tests"
    echo "Testing performance with NIC on NUMA node $NIC_NUMA_NODE"
    echo ""

    # Test 1: Same NUMA node as NIC (expected to be fastest)
    run_network_test $NIC_NUMA_NODE $NIC_NUMA_NODE

    # Test 2: Different NUMA nodes
    run_network_test 0 $NIC_NUMA_NODE
    run_network_test $NIC_NUMA_NODE 0
    run_network_test 1 3
}

run_all_ipc_tests() {
    print_header "Running IPC Socket Tests"
    echo "Testing cross-NUMA memory access performance"
    echo ""

    # Test 1: Same NUMA node (expected to be fastest)
    run_ipc_test 0 0
    run_ipc_test 2 2

    # Test 2: Adjacent NUMA nodes
    run_ipc_test 0 1
    run_ipc_test 2 3

    # Test 3: Distant NUMA nodes
    run_ipc_test 0 2
    run_ipc_test 1 3
    run_ipc_test 0 3
}

usage() {
    echo "Usage: $0 [OPTIONS] [TEST_TYPE]"
    echo ""
    echo "TEST_TYPE:"
    echo "  network    Run network socket tests"
    echo "  ipc        Run IPC socket tests"
    echo "  all        Run all tests (default)"
    echo "  topology   Show NUMA topology only"
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
    echo "  $0 -n 2000 network                   # Run network tests with 2000 samples"
    echo "  $0 -s \"3 512 512\" -b 20 all         # Custom tensor shape and buffer size"
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
