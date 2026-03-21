#!/bin/bash
#
# flash-moe server launcher with auto-restart
#
# Usage:
#   ./serve.sh              # Start 122B 2-bit server (default)
#   ./serve.sh 35b          # Start 35B server (when available)
#   ./serve.sh stop          # Stop server
#   ./serve.sh status        # Check server status
#   ./serve.sh logs          # Tail server logs
#

set -e

# ============================================================================
# Config
# ============================================================================
PORT=8080
MAX_TOKENS=2048
K=8
RESTART_DELAY=3
MAX_RESTARTS=10
RESTART_WINDOW=300  # reset restart counter after 5 minutes of stable running

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFER_DIR="$SCRIPT_DIR/metal_infer"
PID_FILE="$SCRIPT_DIR/.serve.pid"
LOG_FILE="$SCRIPT_DIR/serve.log"

# ============================================================================
# Model configs
# ============================================================================
# Auto-detect model path from HuggingFace cache
find_model_path() {
    local model_id="$1"
    local cache_dir="$HOME/.cache/huggingface/hub/models--${model_id//\//-}"
    if [ -d "$cache_dir/snapshots" ]; then
        ls -1d "$cache_dir/snapshots"/*/ 2>/dev/null | head -1 | sed 's:/$::'
    fi
}

MODEL_122B=$(find_model_path "mlx-community/Qwen3.5-122B-A10B-4bit")

# ============================================================================
# Functions
# ============================================================================

start_server() {
    local model_path="$MODEL_122B"
    local quant_flag="--2bit"
    local model_name="Qwen3.5-122B-A10B"

    if [ -z "$model_path" ]; then
        echo "Error: Model not found in HuggingFace cache."
        echo "Download first: python setup_model.py --model-id mlx-community/Qwen3.5-122B-A10B-4bit"
        exit 1
    fi

    if [ ! -f "$INFER_DIR/infer" ]; then
        echo "Error: $INFER_DIR/infer not found. Run 'make' first."
        exit 1
    fi

    if [ -f "$PID_FILE" ]; then
        local old_pid
        old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "Server already running (PID $old_pid). Use './serve.sh stop' first."
            exit 1
        fi
        rm -f "$PID_FILE"
    fi

    echo "================================================"
    echo "  flash-moe server"
    echo "  Model:  $model_name"
    echo "  Quant:  2-bit experts, K=$K"
    echo "  Port:   $PORT"
    echo "  URL:    http://localhost:$PORT"
    echo "================================================"
    echo ""
    echo "Endpoints:"
    echo "  POST /v1/chat/completions"
    echo "  GET  /v1/models"
    echo "  GET  /health"
    echo ""
    echo "Use in your app:"
    echo "  API Base: http://localhost:$PORT"
    echo "  Model:    qwen3.5-122b"
    echo ""
    echo "Logs: $LOG_FILE"
    echo "Stop: ./serve.sh stop"
    echo ""

    # Auto-restart loop
    local restart_count=0
    local last_start=0

    while true; do
        local now
        now=$(date +%s)

        # Reset counter if stable for RESTART_WINDOW seconds
        if [ $((now - last_start)) -gt $RESTART_WINDOW ]; then
            restart_count=0
        fi

        if [ $restart_count -ge $MAX_RESTARTS ]; then
            echo "[$(date '+%H:%M:%S')] Too many restarts ($MAX_RESTARTS in ${RESTART_WINDOW}s). Giving up." | tee -a "$LOG_FILE"
            rm -f "$PID_FILE"
            exit 1
        fi

        last_start=$now

        if [ $restart_count -gt 0 ]; then
            echo "[$(date '+%H:%M:%S')] Restarting server (attempt $restart_count/$MAX_RESTARTS)..." | tee -a "$LOG_FILE"
            sleep $RESTART_DELAY
        fi

        # Launch inference server
        cd "$INFER_DIR"
        ./infer $quant_flag \
            --model "$model_path" \
            --serve $PORT \
            --k $K \
            --tokens $MAX_TOKENS \
            >> "$LOG_FILE" 2>&1 &

        local server_pid=$!
        echo $server_pid > "$PID_FILE"

        if [ $restart_count -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] Server started (PID $server_pid)" | tee -a "$LOG_FILE"
        fi

        # Wait for server process to exit
        wait $server_pid 2>/dev/null
        local exit_code=$?

        # Check if it was intentionally stopped (SIGTERM = 143)
        if [ $exit_code -eq 143 ] || [ $exit_code -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] Server stopped gracefully (exit $exit_code)" | tee -a "$LOG_FILE"
            rm -f "$PID_FILE"
            exit 0
        fi

        # Crashed — log and restart
        echo "[$(date '+%H:%M:%S')] Server crashed (exit $exit_code)" | tee -a "$LOG_FILE"
        restart_count=$((restart_count + 1))
    done
}

stop_server() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No server running (no PID file)"
        # Also kill any orphaned processes
        pkill -f "infer.*--serve" 2>/dev/null && echo "Killed orphaned server process" || true
        return
    fi

    local pid
    pid=$(cat "$PID_FILE")

    if kill -0 "$pid" 2>/dev/null; then
        # Kill the wrapper (this script) and the infer process
        local infer_pid
        infer_pid=$(pgrep -P "$pid" 2>/dev/null || true)

        kill "$pid" 2>/dev/null
        [ -n "$infer_pid" ] && kill "$infer_pid" 2>/dev/null

        echo "Server stopped (PID $pid)"
    else
        echo "Server not running (stale PID $pid)"
    fi

    rm -f "$PID_FILE"
}

server_status() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Server running (PID $pid)"
            curl -s "http://localhost:$PORT/health" 2>/dev/null && echo "" || echo "  (not responding yet)"
            return 0
        fi
    fi

    # Check for orphaned process
    if pgrep -f "infer.*--serve" >/dev/null 2>&1; then
        echo "Server running (orphaned, no PID file)"
        return 0
    fi

    echo "Server not running"
    return 1
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "No log file yet"
    fi
}

# ============================================================================
# Main
# ============================================================================
case "${1:-start}" in
    start|122b)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 2
        start_server
        ;;
    status)
        server_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
