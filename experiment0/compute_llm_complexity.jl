"""
compute_llm_complexity.jl

Computes LLM-based complexity for pre-generated equations.
Reads equations from JSON and saves complexity results.
"""

using SR_with_LLM_Complexity
using SymbolicRegression: SymbolicRegression as SR
using DynamicExpressions: Expression
using JSON3
using Dates
using Serialization

# Configuration
# Use global NUM_EQUATIONS if set by parent script, otherwise use default
# if !@isdefined(NUM_EQUATIONS)
#     const NUM_EQUATIONS = 10  # Must match the number used in generate_equations.jl
# end

# const NUM_EQUATIONS = 100

const NUM_EQUATIONS = parse(Int, ENV["NUM_EQUATIONS"])

const LLM_MODEL = get(ENV, "LLAMAFILE_MODEL",
                      "Qwen2.5-0.5B-Instruct-Q4_K_M")


const INPUT_DIR = "experimental_results/experiment0"
const OUTPUT_DIR = INPUT_DIR

println("="^60)
println("Computing LLM-Based Complexity")
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


println("\nUsing LLM model: $LLM_MODEL")

# Create ComplexityOptions with LLM enabled
llm_options = ComplexityOptions(
    use_llm_complexity=true,
    model=LLM_MODEL,
    binary_operators=[+, -, *, /],
    unary_operators=[sin, sqrt, exp, log],
    log_complexity_outputs=false  # We'll handle logging manually
)

# Convert Node objects to Expression objects (AbstractExpression type)
println("\nConverting to Expression objects...")
expressions = [Expression(eq; operators=sr_options.operators, variable_names=["x$i" for i in 1:5]) for eq in equations]
println("✓ Successfully converted $(length(expressions)) expressions")

# Compute LLM complexity for each expression
println("\nComputing LLM-based complexity...")
println("  (This will take longer as it calls the LLM for each expression)")
llm_complexities = Int[]
for (i, expr) in enumerate(expressions)
    # Call compute_llm_complexity directly with Expression object
    complexity = SR_with_LLM_Complexity.compute_llm_complexity(expr, llm_options)
    push!(llm_complexities, complexity)
    if i % 10 == 0 || i == length(expressions)
        println("  Processed $i/$(length(expressions)) equations...")
    end
end

# Save LLM complexity results
llm_file = joinpath(OUTPUT_DIR, "$(llm_options.model)_$(NUM_EQUATIONS).json")
open(llm_file, "w") do io
    JSON3.write(io, Dict(
        "timestamp" => timestamp,
        "computed_at" => Dates.format(now(), "yyyymmdd_HHMMSS"),
        "method" => "llm_based",
        "model" => llm_options.model,
        "num_equations" => length(expressions),
        "complexities" => llm_complexities,
        "equations" => equation_strings
    ))
end

println("✓ Saved LLM complexity results to: $llm_file")
println("\nSummary:")
println("  Model used: $(llm_options.model)")
println("  Min complexity: $(minimum(llm_complexities))")
println("  Max complexity: $(maximum(llm_complexities))")
println("  Mean complexity: $(round(sum(llm_complexities) / length(llm_complexities), digits=2))")
