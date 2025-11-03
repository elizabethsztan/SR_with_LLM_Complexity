module Tools

using SymbolicRegression
using SymbolicRegression.PopMemberModule: PopMember
using SymbolicRegression.ComplexityModule: compute_complexity
using DynamicExpressions: Node, string_tree, get_operators
using SymbolicRegression.InterfaceDynamicExpressionsModule: string_variable, string_variable_raw

export string_tree_llm

# ============================================================
# Extended string_tree function with LLM_input option
# ============================================================

"""
    string_tree_llm(tree, options; pretty=false, variable_names=nothing)

Custom string_tree that replaces all constants with "CONST".
This is useful for creating standardized inputs for LLMs.

# Arguments
- `tree`: The expression tree to convert to string
- `options`: Options object containing operators
- `pretty::Bool=false`: Whether to use pretty printing (subscripts, etc.)
- `variable_names`: Custom variable names (optional)
"""
function string_tree_llm(tree, options; variable_names=nothing)
    # Custom constant formatter that always returns "CONST"
    f_constant_llm = (val) -> "C"

    # Get operators from options
    operators = get_operators(tree, options)

    # Choose variable formatter based on pretty flag
    return string_tree(
            tree,
            operators;
            f_variable=string_variable_raw,
            f_constant=f_constant_llm,
            variable_names=variable_names,
        )
end

end # module Tools