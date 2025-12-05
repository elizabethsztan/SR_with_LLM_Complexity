using SR_with_LLM_Complexity
using BenchmarkTools
using Logging
using JSON
using Dates

# Suppress PromptingTools logging (token counts, etc.)
Logging.disable_logging(Logging.Info)
timestamp = Int(floor(Dates.datetime2unix(now())))

"""
    write_results_to_json(filepath, t1, t2, num_populations, num_members_per_population, num_iterations)

Write test results to a JSON file including execution times and experiment parameters.

# Arguments
- `filepath`: Path to the output JSON file
- `t1`: Time taken for test 1 (without LLM complexity)
- `t2`: Time taken for test 2 (with LLM complexity)
- `num_populations`: Number of populations used
- `num_members_per_population`: Number of members per population
- `num_iterations`: Number of iterations
"""
function write_results_to_json(filepath, t1, t2, num_populations, num_members_per_population, num_iterations)
    # Create parent directory if it doesn't exist
    dir = dirname(filepath)
    if !isempty(dir)
        mkpath(dir)
    end

    results = Dict(
        "experiment_parameters" => Dict(
            "num_populations" => num_populations,
            "num_members_per_population" => num_members_per_population,
            "num_iterations" => num_iterations
        ),
        "test1_without_llm" => Dict(
            "time_seconds" => t1
        ),
        "test2_with_llm" => Dict(
            "time_seconds" => t2
        )
    )

    open(filepath, "w") do io
        JSON.print(io, results, 4)  # 4 spaces indentation for readability
    end

    println("Results written to: $filepath")
end

# Create complex test data
# X should be [features, rows] shape in SymbolicRegression
X = randn(2, 100)  # 2 features, 100 samples
y = 2.7 .* exp.(X[1, :] .- 0.2 .* X[2, :]) .* X[2, :] .+ (1 .- X[1, :]) # y = 2.7*exp(x1 - 0.2*x2)*x2^(0.04) + (1 - x1)

num_populations = 1
num_members_per_population = 20
num_iterations = 3

# Test 1: Using ComplexityOptions WITHOUT LLM complexity (should use default)
println("Test 1: ComplexityOptions with LLM disabled")

options1 = ComplexityOptions(
    use_llm_complexity=false,
    binary_operators=[+, *, -, /],
    unary_operators=[sin, cos],
    populations=num_populations,
    population_size=num_members_per_population, 
    log_complexity_outputs=true, 
    log_standard_file_path="experimental_results/complex_expression/standard_complexity_log_$(timestamp).json"
)

println("Starting symbolic regression WITHOUT LLM complexity...")
t1 = @elapsed hall_of_fame1 = equation_search(X, y; niterations=num_iterations, options=options1)
println("Time taken: $t1 seconds")
println()

# Test 2: Using ComplexityOptions WITH LLM complexity
println("Test 2: ComplexityOptions with LLM enabled")

options2 = ComplexityOptions(
    use_llm_complexity=true,
    binary_operators=[+, *, -, /],
    unary_operators=[sin, cos],
    populations=num_populations,
    population_size=num_members_per_population, 
    log_complexity_outputs=true, 
    log_llm_file_path="experimental_results/complex_expression/llm_complexity_log_$(timestamp).json"
)

# Run with LLM complexity enabled
println("Starting symbolic regression WITH LLM-based complexity...")
t2 = @elapsed hall_of_fame2 = equation_search(X, y; niterations=num_iterations, options=options2)
println("Time taken: $t2 seconds")
println()

# Write results to JSON file
output_file = "experimental_results/complex_expression/complex_test_results_$(timestamp).json"
write_results_to_json(output_file, t1, t2, num_populations, num_members_per_population, num_iterations)