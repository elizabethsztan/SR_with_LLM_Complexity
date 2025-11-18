"""
LLMComplexityOptions.jl

Constructor and initialization logic for ComplexityOptions.
"""

module LLMComplexityOptionsModule

using SymbolicRegression
using SymbolicRegression: Options
using ..LLMComplexityOptionsStructModule:
    LLMComplexityOptions, LLM_COMPLEXITY_OPTIONS_KEYS
import ..LLMComplexityOptionsStructModule: ComplexityOptions
using ..LLMComplexity: initialize_log

export ComplexityOptions

"""
    ComplexityOptions(; kws...)

Create a ComplexityOptions object that combines LLM complexity options with
SymbolicRegression.jl options.

# LLM Complexity Options
- `use_llm_complexity::Bool=false`: Enable LLM-based complexity evaluation
- `user_examples::String`: Examples to guide complexity assignment (default: "x1 + x2 + C has complexity 3, C * sin(x1) has complexity 4, sin(sin(sin(x1))) has complexity 10")

# SymbolicRegression.jl Options
All standard SymbolicRegression.jl options are also supported. See `SymbolicRegression.Options`
for the full list.

# Examples
```julia
# Basic usage with LLM complexity
options = ComplexityOptions(
    use_llm_complexity=true,
    user_examples="x1 + x2 has complexity 2, sin(x1) has complexity 3",
    parsimony=0.01,
    maxsize=30
)

# Access options transparently
println(options.use_llm_complexity)  # true
println(options.parsimony)            # 0.01
```
"""
function ComplexityOptions(;
    # LLM Complexity specific options
    use_llm_complexity::Bool=false,
    user_examples::String=LLMComplexityOptions().user_examples,  # Use default from struct
    log_complexity_outputs::Bool=false,
    log_llm_file_path::String=LLMComplexityOptions().log_llm_file_path,
    log_standard_file_path::String=LLMComplexityOptions().log_standard_file_path,
    # All other kwargs go to SymbolicRegression.Options
    kws...
)
    # Create LLM complexity options
    llm_complexity_options = LLMComplexityOptions(
        use_llm_complexity=use_llm_complexity,
        user_examples=user_examples,
        log_complexity_outputs=log_complexity_outputs,
        log_llm_file_path=log_llm_file_path,
        log_standard_file_path=log_standard_file_path
    )

    # Initialize log files if logging is enabled
    if log_complexity_outputs
        if use_llm_complexity
            initialize_log(log_llm_file_path)
        else
            initialize_log(log_standard_file_path)
        end
    end

    # Create SymbolicRegression options (pass all remaining kwargs)
    sr_options = Options(; kws...)

    # Create composite options
    return ComplexityOptions{typeof(sr_options)}(llm_complexity_options, sr_options)
end

end # module
