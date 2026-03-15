#!/usr/bin/env bash
#
# setup_llamacpp.sh — clone and build llama.cpp for prefill-decode-bench.
#
# Run once:   ./setup_llamacpp.sh
# Update:     ./setup_llamacpp.sh --update
#
# Produces:   vendor/llama.cpp/build/bin/llama-bench
#             vendor/llama.cpp/build/bin/llama-cli
#
set -euo pipefail

VENDOR_DIR="$(cd "$(dirname "$0")" && pwd)/vendor"
LLAMA_DIR="$VENDOR_DIR/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"

CMAKE_EXTRA_ARGS=""

# Auto-detect platform and set GPU backend
case "$(uname -s)" in
    Darwin)
        CMAKE_EXTRA_ARGS="-DGGML_METAL=ON"
        echo "Platform: macOS — enabling Metal"
        ;;
    Linux)
        if command -v nvcc &>/dev/null; then
            CMAKE_EXTRA_ARGS="-DGGML_CUDA=ON"
            echo "Platform: Linux — enabling CUDA"
        else
            echo "Platform: Linux — CPU only (install CUDA toolkit for GPU support)"
        fi
        ;;
    *)
        echo "Platform: $(uname -s) — CPU only"
        ;;
esac

if [ "${1:-}" = "--update" ] && [ -d "$LLAMA_DIR" ]; then
    echo "Updating llama.cpp..."
    cd "$LLAMA_DIR"
    git pull --ff-only
    echo "Rebuilding ($JOBS threads)..."
    cmake --build "$BUILD_DIR" --config Release -j "$JOBS"
    echo ""
    echo "Updated. Binary: $BUILD_DIR/bin/llama-bench"
    exit 0
fi

if [ -f "$BUILD_DIR/bin/llama-bench" ]; then
    echo "llama-bench already built at: $BUILD_DIR/bin/llama-bench"
    echo "Run with --update to pull latest and rebuild."
    exit 0
fi

mkdir -p "$VENDOR_DIR"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
else
    echo "llama.cpp already cloned, pulling latest..."
    cd "$LLAMA_DIR" && git pull --ff-only && cd -
fi

echo ""
echo "Configuring (Metal=$([[ "$CMAKE_EXTRA_ARGS" == *METAL* ]] && echo ON || echo OFF), CUDA=$([[ "$CMAKE_EXTRA_ARGS" == *CUDA* ]] && echo ON || echo OFF))..."
cmake -B "$BUILD_DIR" -S "$LLAMA_DIR" $CMAKE_EXTRA_ARGS

echo ""
echo "Building ($JOBS threads)..."
cmake --build "$BUILD_DIR" --config Release -j "$JOBS"

echo ""
if [ -f "$BUILD_DIR/bin/llama-bench" ]; then
    echo "Build successful."
    echo "  llama-bench: $BUILD_DIR/bin/llama-bench"
    echo "  llama-cli:   $BUILD_DIR/bin/llama-cli"
    echo ""
    echo "Now run:"
    echo "  python profiler.py --backend llamacpp --model path/to/model.gguf"
else
    echo "Build failed — llama-bench binary not found."
    exit 1
fi
