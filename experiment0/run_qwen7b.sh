#!/bin/bash
#SBATCH --job-name=qwen7b_sr
#SBATCH --output=logs/qwen7b_sr_%j.out
#SBATCH --error=logs/qwen7b_sr_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Load Julia module (check available versions with: module avail julia)
module load julia

# Set environment variables
export LLAMAFILE_MODEL="Qwen2.5-7B-Instruct-1M-Q6_K"
export LLM_PORT=11449
export LLM_FLAGS="--gpu auto"

# Number of equations to process (must match generate_equations.jl)
export NUM_EQUATIONS=100

# Use SLURM_SUBMIT_DIR (directory where sbatch was called from)
# If not set (running locally), fall back to script directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"/..
fi
PROJECT_DIR="$(pwd)"

echo "=================================="
echo "Running Full Pipeline on HPC"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Project Dir: $PROJECT_DIR"
echo "=================================="

# Start the LLM server directly (not through Julia)
echo "Starting LLM server..."
echo "Server output will be logged to: logs/server_${SLURM_JOB_ID}.log"

LLAMAFILE_PATH="llamafiles/Qwen2.5-7B-Instruct-1M-Q6_K.llamafile"
chmod +x $LLAMAFILE_PATH

# Start llamafile directly in background with GPU offloading
./$LLAMAFILE_PATH --server --nobrowser --port ${LLM_PORT} -ngl 9999 > logs/server_${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!

# Give it time to load the model (takes ~5 seconds)
sleep 10

# Check if the server process is still running
if ! ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "ERROR: Server process died immediately after starting!"
    echo "Check logs/server_${SLURM_JOB_ID}.log for details"
    cat logs/server_${SLURM_JOB_ID}.log
    exit 1
fi

echo "Server process started (PID: $SERVER_PID)"

# Function to cleanup server on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping LLM server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
    fi
    echo "Done."
}

# Register cleanup function to run on script exit
trap cleanup EXIT INT TERM

# Wait for server to be ready
echo "Waiting for LLM server to be ready..."
MAX_WAIT=300  # Maximum 5 minutes
WAIT_TIME=0
while ! nc -z localhost $LLM_PORT 2>/dev/null; do
    # Check if server process is still running
    if ! ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "ERROR: Server process terminated unexpectedly!"
        echo "Last 50 lines of server log:"
        tail -50 logs/server_${SLURM_JOB_ID}.log
        exit 1
    fi

    if [ $WAIT_TIME -ge $MAX_WAIT ]; then
        echo "ERROR: LLM server did not start within $MAX_WAIT seconds"
        echo "Last 50 lines of server log:"
        tail -50 logs/server_${SLURM_JOB_ID}.log
        exit 1
    fi

    echo "Waiting for server on port $LLM_PORT... ($WAIT_TIME/$MAX_WAIT seconds)"

    # Show server log every 30 seconds
    if [ $((WAIT_TIME % 30)) -eq 0 ] && [ $WAIT_TIME -gt 0 ]; then
        echo "--- Server log tail ---"
        tail -10 logs/server_${SLURM_JOB_ID}.log
        echo "--- End log tail ---"
    fi

    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
done

echo "✓ LLM server is ready (PID: $SERVER_PID)!"

# Test if server is actually responding to requests
echo "Testing server responsiveness..."
if curl -s http://localhost:${LLM_PORT}/health > /dev/null 2>&1 || curl -s http://localhost:${LLM_PORT}/ > /dev/null 2>&1; then
    echo "✓ Server is responding to requests"
else
    echo "WARNING: Server is listening but not responding to HTTP requests"
    echo "Check logs/server_${SLURM_JOB_ID}.log for details"
fi
echo ""

# Run the LLM complexity computation
echo "=================================="
echo "Running LLM complexity computation on Qwen7B"
echo "=================================="
echo "Processing $NUM_EQUATIONS equations"
echo ""

# Run the Julia script with NUM_EQUATIONS as environment variable
julia --project=. experiment0/compute_llm_complexity.jl

TEST1_EXIT=$?
if [ $TEST1_EXIT -ne 0 ]; then
    echo "WARNING: LLM complexity computation failed with exit code $TEST1_EXIT"
fi

echo ""
echo "=================================="
echo "All tests completed!"
echo "=================================="

# Cleanup will happen automatically via trap