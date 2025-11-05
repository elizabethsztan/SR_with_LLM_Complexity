using SR_with_LLM_Complexity
using BenchmarkTools
using Logging

# Suppress PromptingTools logging (token counts, etc.)
Logging.disable_logging(Logging.Info)

# Create simple test data
# X should be [features, rows] shape in SymbolicRegression
X = randn(2, 100)  # 2 features, 100 samples
y = X[1, :] .^ 2 .+ X[2, :]  # y = x1^2 + x2

# Test 1: Using ComplexityOptions WITHOUT LLM complexity (should use default)
println("=" ^ 60)
println("Test 1: ComplexityOptions with LLM disabled")
println("=" ^ 60)

options1 = ComplexityOptions(
    use_llm_complexity=false,
    binary_operators=[+, *, -, /],
    unary_operators=[sin, cos],
    populations=1,
    population_size=20
)

println("use_llm_complexity: ", options1.use_llm_complexity)
println("maxsize: ", options1.maxsize)
println("Starting symbolic regression WITHOUT LLM complexity...")
t1 = @elapsed hall_of_fame1 = equation_search(X, y; niterations=3, options=options1)
println("Time taken: $t1 seconds")
println()

# Test 2: Using ComplexityOptions WITH LLM complexity
println("=" ^ 60)
println("Test 2: ComplexityOptions with LLM enabled")
println("=" ^ 60)

options2 = ComplexityOptions(
    use_llm_complexity=true,
    user_examples="x1 has complexity 1, x1 + x2 has complexity 2, sin(x1) has complexity 5",
    binary_operators=[+, *, -, /],
    unary_operators=[sin, cos],
    populations=1,
    population_size=20
)

println("use_llm_complexity: ", options2.use_llm_complexity)
println("user_examples: ", options2.user_examples)
println()

# Run with LLM complexity enabled
println("Starting symbolic regression WITH LLM-based complexity...")
t2 = @elapsed hall_of_fame2 = equation_search(X, y; niterations=3, options=options2)
println("Time taken: $t2 seconds")

println("\n" * "=" ^ 60)
println("Results:")
println("=" ^ 60)
println("\nTest 1 (no LLM) - Best expressions:")
println(hall_of_fame1)
println("\nTest 2 (with LLM) - Best expressions:")
println(hall_of_fame2)

println("\n" * "=" ^ 60)
println("Timing Comparison:")
println("=" ^ 60)
println("Time without LLM: $t1 seconds")
println("Time with LLM: $t2 seconds")
println("Difference: $(t2 - t1) seconds")