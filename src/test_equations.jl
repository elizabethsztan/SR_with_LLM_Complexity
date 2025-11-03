using SymbolicRegression
using SymbolicRegression.PopMemberModule: PopMember
using DynamicExpressions: Node, string_tree, get_operators
using SymbolicRegression.InterfaceDynamicExpressionsModule: string_variable, string_variable_raw
using PromptingTools
using PromptingTools: SystemMessage, UserMessage, CustomOpenAISchema
using JSON  # Move this to the top level

# Import custom functions from tools.jl
include("tools.jl")

# Define options
options = Options(
    binary_operators=[+, -, *, /, ^],
    unary_operators=[sin, cos, exp]
)

# Create dataset
X = rand(Float64, 2, 100)  # 2 features, 100 samples
y = sin.(X[1, :]) .+ X[2, :] .+ 3.0
dataset = Dataset(X, y)

# 1. PopMember for sin(x1) + x2 + 3.0 (initialized with dataset)
x1_node = Node(Float64; feature=1)
x2_node = Node(Float64; feature=2)
three_node = Node(Float64; val=3.0)

sin_x1 = Node(1, x1_node)  # sin(x1)
sum1 = Node(1, sin_x1, x2_node)  # sin(x1) + x2
expr_tree = Node(1, sum1, three_node)  # (sin(x1) + x2) + 3.0

pop_member1 = PopMember(
    dataset,
    expr_tree,
    options;
    deterministic=true
)

# 2. PopMember for exp(x^x) - also use dataset constructor for simplicity
x_node = Node(Float64; feature=1)
x_pow_x = Node(5, x_node, x_node)  # x^x
exp_x_pow_x = Node(3, x_pow_x)  # exp(x^x)

pop_member2 = PopMember(
    dataset,
    exp_x_pow_x,
    options;
    deterministic=true
)

println("PopMember 1")
println(string_tree_llm(pop_member1.tree, options))

println("PopMember 2")
println(string_tree_llm(pop_member2.tree, options))



#NOW try to call the LLM and feed in my equations


function simple_llm_call(expression::PopMember, options; user_examples::String="")

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

simple_llm_call(pop_member2, options)