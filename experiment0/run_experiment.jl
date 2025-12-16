"""
run_experiment.jl

Master script that runs the complete experiment pipeline:
1. Generate random equations
2. Compute standard complexity
3. Compute LLM complexity
"""

println("="^60)
println("Running Complete Experiment Pipeline")
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
println("\nGenerated files:")
println("  - equations_$(NUM_EQUATIONS).json")
println("  - standard_complexity_$(NUM_EQUATIONS).json")
println("  - llm_complexity_$(NUM_EQUATIONS).json")
