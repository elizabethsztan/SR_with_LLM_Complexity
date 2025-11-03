"""
LLMComplexity.jl

Extends SymbolicRegression.jl's Complexity module.
"""
module LLMComplexity

using SymbolicRegression
using DynamicExpressions
using PromptingTools
using PromptingTools: SystemMessage, UserMessage, aigenerate, CustomOpenAISchema

# Import your own string_tree_llm function
include("tools.jl")
using .Tools: string_tree_llm

# Import to extend (use `import` not `using` when you want to add methods)
import SymbolicRegression: compute_complexity

function evaluate_expression(expression::AbstractExpression, options::AbstractOptions; user_examples::String="")

    expression_string = string_tree_llm(expression.tree, options)

    examples = user_examples != "" ? user_examples : "x1 + x2 + C has complexity 3, C * sin(x1) has complexity 4, sin(sin(sin(x1))) has complexity 10"

    constraints = "All constants are represented with the symbol C."

    system_msg = SystemMessage("You are a helpful assistant that assigns complexity values to mathematical expressions depending on how human-interpretable they are. Higher scores indicate higher complexities and lower interpretability. Example mappings include $examples. Your only output should be the complexity value.")

    user_msg = UserMessage("Provide an integer value for the complexity of the following expression: $expression_string. $constraints.")

    conversation = [system_msg, user_msg]


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
    println(typeof(Int(complexity)))

    return complexity
end


#pass in a PopMember 
function compute_complexity(
    tree::AbstractExpression, options::AbstractOptions; break_sharing=Val(false)
)

    complexity = evaluate_expression(tree, options)
    return complexity
end

end


