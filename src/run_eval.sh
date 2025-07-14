#!/bin/bash

# SVGenius Evaluation Runner
# This script runs comprehensive evaluations across different SVG processing tasks
# Author: SVGenius Team
# Date: July 14, 2025

set -e  # Exit on any error

# Configuration
MODEL="mock-llm"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="evaluation_results_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run evaluation with status reporting
run_evaluation() {
    local module=$1
    local input_file=$2
    local output_file=$3
    local description=$4
    
    print_status "Running $description..."
    if python -m "$module" --input "$input_file" --output "${RESULTS_DIR}/$output_file" --model "$MODEL"; then
        print_success "Completed $description"
    else
        print_error "Failed $description"
        return 1
    fi
}

# Create results directory
print_status "Creating results directory: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# Print header
echo "========================================"
echo "      SVGenius Evaluation Suite        "
echo "========================================"
echo "Model: $MODEL"
echo "Results Directory: $RESULTS_DIR"
echo "Start Time: $(date)"
echo "========================================"


# ==============================================================================
# TEXT-TO-SVG GENERATION EVALUATIONS
# ==============================================================================
print_status "Starting Text-to-SVG Generation Evaluations..."

run_evaluation "generation.text2svg.evaluation" \
    "tasks/generation/text_svg/easy_svg_captions.json" \
    "text2svg_easy_svg_captions_results.json" \
    "Text2SVG - Easy Captions"

run_evaluation "generation.text2svg.evaluation" \
    "tasks/generation/text_svg/medium_svg_captions.json" \
    "text2svg_medium_svg_captions_results.json" \
    "Text2SVG - Medium Captions"

run_evaluation "generation.text2svg.evaluation" \
    "tasks/generation/text_svg/hard_svg_captions.json" \
    "text2svg_hard_svg_captions_results.json" \
    "Text2SVG - Hard Captions"


# ==============================================================================
# BUG FIXING EVALUATIONS
# ==============================================================================
print_status "Starting Bug Fixing Evaluations..."

run_evaluation "editing.bug_fixing.evaluation" \
    "tasks/editing/bug_fixing/easy_svg_errors_dataset.json" \
    "bug_fixing_easy_svg_errors_results.json" \
    "Bug Fixing - Easy SVG Errors"

run_evaluation "editing.bug_fixing.evaluation" \
    "tasks/editing/code_optimization/medium_svg_optimization_results.json" \
    "bug_fixing_medium_svg_optimization_results.json" \
    "Bug Fixing - Medium SVG Optimization"

run_evaluation "editing.bug_fixing.evaluation" \
    "tasks/editing/bug_fixing/hard_errors_dataset.json" \
    "bug_fixing_hard_errors_results.json" \
    "Bug Fixing - Hard Errors"

# ==============================================================================
# CODE OPTIMIZATION EVALUATIONS
# ==============================================================================
print_status "Starting Code Optimization Evaluations..."

run_evaluation "editing.code_opti.evaluation" \
    "tasks/editing/code_optimization/easy_svg_optimization_results.json" \
    "code_opti_easy_svg_optimization_results.json" \
    "Code Optimization - Easy SVG"

run_evaluation "editing.code_opti.evaluation" \
    "tasks/editing/code_optimization/medium_svg_optimization_results.json" \
    "code_opti_medium_svg_optimization_results.json" \
    "Code Optimization - Medium SVG"

run_evaluation "editing.code_opti.evaluation" \
    "tasks/editing/code_optimization/hard_svg_optimization_results.json" \
    "code_opti_hard_svg_optimization_results.json" \
    "Code Optimization - Hard SVG"

# ==============================================================================
# UNDERSTANDING EVALUATIONS (Currently disabled)
# ==============================================================================
print_warning "Understanding evaluations are currently disabled"
# Uncomment the lines below to enable understanding evaluations
#
# print_status "Starting Understanding Evaluations..."
# run_evaluation "understanding.und.evaluation_fix" \
#     "tasks/understanding/easy_generation_results.json" \
#     "understanding_easy_generation_results.json" \
#     "Understanding - Easy Generation"
#
# run_evaluation "understanding.evaluation_fix" \
#     "tasks/understanding/medium_generation_results.json" \
#     "understanding_medium_generation_results.json" \
#     "Understanding - Medium Generation"
#
# run_evaluation "understanding.evaluation_fix" \
#     "tasks/understanding/hard_generation_results.json" \
#     "understanding_hard_generation_results.json" \
#     "Understanding - Hard Generation"

# ==============================================================================
# COMPLETION SUMMARY
# ==============================================================================
echo "========================================"
print_success "All evaluations completed successfully!"
echo "Results saved in: $RESULTS_DIR"
echo "End Time: $(date)"

# List all result files
print_status "Generated result files:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || print_warning "No result files found"

echo "========================================"
