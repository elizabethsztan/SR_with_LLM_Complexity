"""
LLMComplexity.jl

"""
module LLMComplexity

using SymbolicRegression
using SymbolicRegression: AbstractOptions, AbstractExpression
using DynamicExpressions
using PromptingTools
using PromptingTools: SystemMessage, UserMessage, aigenerate, CustomOpenAISchema
using JSON3

# Global lock for thread-safe logging
const LOG_LOCK = ReentrantLock()

# Import your own string_tree_llm function
include("Tools.jl")
using .Tools: string_tree_llm

# Import our ComplexityOptions type
using ..LLMComplexityOptionsStructModule: ComplexityOptions

# Export the core LLM complexity function for use by the patching module
export compute_llm_complexity

"""
    initialize_log(log_file_path::String)

Creates a log file with an empty JSON array if it doesn't exist.
If it exists, does nothing (preserves existing data).
Creates parent directories if they don't exist.
"""
function initialize_log(log_file_path::String)
    # Create parent directory if it doesn't exist
    dir = dirname(log_file_path)
    if !isempty(dir)
        mkpath(dir)
    end

    # Only create file if it doesn't exist
    if !isfile(log_file_path)
        open(log_file_path, "w") do io
            JSON3.write(io, Int[])
        end
    end
end

"""
    log_complexity(complexity::Int, log_file_path::String)

Appends a complexity value to a JSON array file in a thread-safe manner.
Assumes the log file has been initialized with initialize_log().
"""
function log_complexity(complexity::Int, log_file_path::String)
    lock(LOG_LOCK) do
        # Read existing log (file must be initialized first)
        complexities = JSON3.read(read(log_file_path, String), Vector{Int})

        # Append new complexity and write back
        push!(complexities, complexity)
        open(log_file_path, "w") do io
            JSON3.write(io, complexities)
        end
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


    # response = aigenerate(
    #     CustomOpenAISchema(),
    #     conversation;
    #     api_key="local-server",
    #     model="qwen3-coder-30b-a3b-instruct",
    #     api_kwargs=(url="http://localhost:11449/v1", max_tokens=200, temperature = 0.0),
    #     http_kwargs=(retries=3, readtimeout=60)
    # )

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

end # module LLMComplexity


