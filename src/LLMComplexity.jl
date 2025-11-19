"""
LLMComplexity.jl

Extends SymbolicRegression.jl's Complexity module.
"""
module LLMComplexity

# Opt-out of precompilation to allow method overwriting
__precompile__(false)

using SymbolicRegression
using SymbolicRegression: AbstractOptions, AbstractExpression
using DynamicExpressions
using PromptingTools
using PromptingTools: SystemMessage, UserMessage, aigenerate, CustomOpenAISchema
using JSON3

# Import your own string_tree_llm function
include("Tools.jl")
using .Tools: string_tree_llm

# Import to extend (use `import` not `using` when you want to add methods)
import SymbolicRegression.ComplexityModule: compute_complexity

# Import our ComplexityOptions type
using ..LLMComplexityOptionsStructModule: ComplexityOptions

"""
    initialize_log(log_file_path::String)

Creates or overwrites a log file with an empty JSON array.
Creates parent directories if they don't exist.
"""
function initialize_log(log_file_path::String)
    # Create parent directory if it doesn't exist
    dir = dirname(log_file_path)
    if !isempty(dir)
        mkpath(dir)
    end

    open(log_file_path, "w") do io
        JSON3.write(io, Int[])
    end
end

"""
    log_complexity(complexity::Int, log_file_path::String)

Appends a complexity value to a JSON array file.
"""
function log_complexity(complexity::Int, log_file_path::String)
    # Read existing log or create new array
    complexities = if isfile(log_file_path)
        try
            JSON3.read(read(log_file_path, String), Vector{Int})
        catch
            Int[]
        end
    else
        Int[]
    end

    # Append new complexity
    push!(complexities, complexity)

    # Write back to file
    open(log_file_path, "w") do io
        JSON3.write(io, complexities)
    end
end

function compute_llm_complexity(expression_tree::AbstractExpression, options)

    expression_string = string_tree_llm(expression_tree, options)

    # println("Expression string: $expression_string (tree object_id: ", objectid(expression_tree), ")")

    # Get examples from options (always available via ComplexityOptions)
    examples = options.user_examples

    constraints = "All constants are represented with the symbol C."

    system_msg = SystemMessage("You are a helpful assistant that assigns complexity values to mathematical expressions depending on how human-interpretable they are. Higher scores indicate higher complexities and lower interpretability. Example mappings include $examples. Your output should be the complexity value ONLY. DO NOT output anything except an integer complexity value.")

    user_msg = UserMessage("Provide an integer value for the complexity of the following expression: $expression_string. $constraints.")

    conversation = [system_msg, user_msg]


    # Use gpt-5

    # response = aigenerate(
    #     CustomOpenAISchema(),
    #     conversation;
    #     api_key="local-server",
    #     model="gpt-5",
    #     api_kwargs=(url="http://127.0.0.1:8000/v1", max_tokens=200, temperature = 0.0),
    #     http_kwargs=(retries=3, readtimeout=60)
    # )

    # output = response.content
    # complexity = parse(Int64, output)


    # Use Qwen2.5-0.5B-Instruct

    response = aigenerate(
        CustomOpenAISchema(),
        conversation;
        api_key="local-server",
        model="Qwen2.5-0.5B-Instruct-Q4_K_M",
        api_kwargs=(url="http://localhost:11449/v1", max_tokens=200, temperature = 0.0),
        http_kwargs=(retries=3, readtimeout=60)
    )

    output = response.content
    complexity = parse(Int64, output)

    # Log complexity if enabled
    if options.log_complexity_outputs
        log_complexity(complexity, options.log_llm_file_path)
    end

    # println("Complexity: $complexity")

    return complexity
end

# Extends original compute_complexity
function compute_complexity(
    tree::AbstractExpression,
    options::ComplexityOptions;  # Note: ComplexityOptions, not AbstractOptions
    break_sharing=Val(false)
)
    if options.use_llm_complexity
        # Use LLM-based complexity evaluation
        return compute_llm_complexity(tree, options)
    else
        # Call the original compute_complexity by passing options.sr_options (type Options)
        # Julia will dispatch to the original SymbolicRegression method because
        # options.sr_options is type Options, not ComplexityOptions
        complexity = compute_complexity(tree, options.sr_options; break_sharing=break_sharing)

        # Log standard complexity if enabled
        if options.log_complexity_outputs
            log_complexity(complexity, options.log_standard_file_path)
        end

        return complexity
    end
end

end


