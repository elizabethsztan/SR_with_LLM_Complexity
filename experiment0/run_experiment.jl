"""
run_experiment.jl

Master script that runs the complete experiment pipeline:
1. Generate random equations
2. Compute standard complexity
3. Compute LLM complexity

Usage:
  julia --project=. experiment0/run_experiment.jl [num_equations] [model_name]

Arguments:
  num_equations - Number of equations to generate (default: 10)
  model_name    - LLM model to use (default: Qwen2.5-0.5B-Instruct-Q4_K_M)

Examples:
  julia --project=. experiment0/run_experiment.jl 100
  julia --project=. experiment0/run_experiment.jl 100 Qwen2.5-7B-Instruct-1M-llamafile
"""

# Parse command-line arguments
if length(ARGS) >= 1
    global NUM_EQUATIONS = parse(Int, ARGS[1])
else
    global NUM_EQUATIONS = 10  # Default
end

if length(ARGS) >= 2
    global LLM_MODEL = ARGS[2]
else
    global LLM_MODEL = "Qwen2.5-0.5B-Instruct-Q4_K_M"  # Default model
end

println("="^60)
println("Running Complete Experiment Pipeline")
println("="^60)
println("Number of equations: $NUM_EQUATIONS")
println("LLM Model: $LLM_MODEL")
println("="^60)

# Step 1: Generate equations
println("\n" * "="^60)
println("STEP 1/3: Generating Random Equations")
println("="^60)
include("generate_equations.jl")

# Step 2: Compute standard complexity
println("\n" * "="^60)
println("STEP 2/3: Computing Standard Complexity")
println("="^60)
include("compute_standard_complexity.jl")

# Step 3: Compute LLM complexity
println("\n" * "="^60)
println("STEP 3/3: Computing LLM Complexity")
println("="^60)
include("compute_llm_complexity.jl")

# Final summary
println("\n" * "="^60)
println("EXPERIMENT COMPLETE")
println("="^60)
println("\nAll results saved to: experimental_results/experiment0/")
