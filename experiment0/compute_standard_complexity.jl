"""
compute_standard_complexity.jl

Computes standard complexity (tree counting) for pre-generated equations.
Reads equations from JSON and saves complexity results.
"""

using SR_with_LLM_Complexity
using SymbolicRegression: SymbolicRegression as SR
using DynamicExpressions: Node
using JSON3
using Dates
using Serialization

# Configuration
const NUM_EQUATIONS = 10  # Must match the number used in generate_equations.jl
const INPUT_DIR = "experimental_results/experiment0"
const OUTPUT_DIR = INPUT_DIR

println("="^60)
println("Computing Standard Complexity (Tree Counting)")
println("="^60)

# Load equations from serialized file
equations_binary_file = joinpath(INPUT_DIR, "equations_$(NUM_EQUATIONS).jls")
if !isfile(equations_binary_file)
    error("Equations file not found: $equations_binary_file\nPlease run generate_equations.jl first!")
end

println("\nLoading equations from: $equations_binary_file")
data = deserialize(equations_binary_file)
equations = data.equations
sr_options = data.sr_options
println("✓ Loaded $(length(equations)) equations")

# Also load the JSON file for metadata and equation strings
equations_json_file = joinpath(INPUT_DIR, "equations_$(NUM_EQUATIONS).json")
equations_data = JSON3.read(read(equations_json_file, String))
equation_strings = equations_data["equations"]
timestamp = equations_data["timestamp"]

# Compute standard complexity for each equation
println("\nComputing standard complexity (tree counting)...")
standard_complexities = Int[]
for (i, eq) in enumerate(equations)
    complexity = SR.compute_complexity(eq, sr_options)
    push!(standard_complexities, complexity)
    if i % 10 == 0 || i == length(equations)
        println("  Processed $i/$(length(equations)) equations...")
    end
end

# Save standard complexity results
standard_file = joinpath(OUTPUT_DIR, "standard_complexity_$(NUM_EQUATIONS).json")
open(standard_file, "w") do io
    JSON3.write(io, Dict(
        "timestamp" => timestamp,
        "computed_at" => Dates.format(now(), "yyyymmdd_HHMMSS"),
        "method" => "standard_tree_counting",
        "num_equations" => length(equations),
        "complexities" => standard_complexities,
        "equations" => equation_strings
    ))
end

println("✓ Saved standard complexity results to: $standard_file")
println("\nSummary:")
println("  Min complexity: $(minimum(standard_complexities))")
println("  Max complexity: $(maximum(standard_complexities))")
println("  Mean complexity: $(round(sum(standard_complexities) / length(standard_complexities), digits=2))")
