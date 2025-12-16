"""
generate_equations.jl

Generates random equations and saves them to JSON.
This is the parent script that generates equations for complexity evaluation.
"""

using SR_with_LLM_Complexity
using SymbolicRegression: SymbolicRegression as SR
using JSON3
using Dates
using Serialization

# Include the equation generator from src
include("../src/gen_equations.jl")

# # Configuration
# # Use global NUM_EQUATIONS if set by parent script, otherwise use default
# if !@isdefined(NUM_EQUATIONS)
#     const NUM_EQUATIONS = 10  # Number of random expressions to generate
# end
# # Scale MAX_ATTEMPTS based on NUM_EQUATIONS (typically need ~100x attempts per equation)
# const MAX_ATTEMPTS = max(10000, NUM_EQUATIONS * 150)  # Maximum attempts to generate equations
const OUTPUT_DIR = "experimental_results/experiment0"

NUM_EQUATIONS = 100
MAX_ATTEMPTS = 15000

println("="^60)
println("Generating Random Equations Dataset")
println("="^60)

# Create output directory
if !isdir(OUTPUT_DIR)
    mkpath(OUTPUT_DIR)
    println("✓ Created output directory: $OUTPUT_DIR")
end

# Generate timestamp for unique filenames
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

# Generate random expressions using gen_equations
println("\nGenerating $NUM_EQUATIONS random expressions...")
equations, sr_options, sample_X = generate_equations(
    num_equations=NUM_EQUATIONS,
    max_attempts=MAX_ATTEMPTS,
    binary_operators=(+, -, *, /),
    unary_operators=(sin, sqrt, exp, log),
    max_num_features=5
)

println("✓ Successfully generated $(length(equations)) equations")

# Convert equations to strings for display/JSON storage
equation_strings = [SR.string_tree(eq, sr_options) for eq in equations]

# Save equation strings and metadata to JSON
equations_json_file = joinpath(OUTPUT_DIR, "equations_$(NUM_EQUATIONS).json")
open(equations_json_file, "w") do io
    JSON3.write(io, Dict(
        "timestamp" => timestamp,
        "num_equations" => length(equations),
        "equations" => equation_strings,
        "binary_operators" => string.([+, -, *, /]),
        "unary_operators" => string.([sin, sqrt, exp, log]),
        "max_num_features" => 5
    ))
end
println("✓ Saved equation strings to: $equations_json_file")

# Save raw equation objects (Node objects) using serialization for later use
equations_binary_file = joinpath(OUTPUT_DIR, "equations_$(NUM_EQUATIONS).jls")
serialize(equations_binary_file, (equations=equations, sr_options=sr_options, sample_X=sample_X))
println("✓ Saved equation objects to: $equations_binary_file")
