"""
ComplexityPatching.jl

Runtime patching system for SymbolicRegression.jl's compute_complexity function.
This allows precompilation to work by deferring method redefinition to runtime via __init__().
"""
module ComplexityPatchingModule

using SymbolicRegression: SymbolicRegression as SR
using SymbolicRegression: AbstractExpression, AbstractOptions, DATA_TYPE, LOSS_TYPE, ComplexityMapping
using DynamicExpressions: AbstractExpressionNode, get_tree, count_nodes

# Import from parent module (will be set during include)
using ..LLMComplexityOptionsStructModule: ComplexityOptions, LLMComplexityOptions

# Global state for LLM complexity
global const ACTIVE_LLM_OPTIONS = Ref{Union{ComplexityOptions,Nothing}}(nothing)
global const LLM_COMPLEXITY_FUNC = Ref{Union{Function,Nothing}}(nothing)
global const STANDARD_COMPLEXITY_LOG_FUNC = Ref{Union{Function,Nothing}}(nothing)

export set_active_llm_options, clear_active_llm_options, install_patches!, set_llm_complexity_function, set_standard_complexity_log_function

"""
    set_active_llm_options(options::ComplexityOptions)

Set the global ComplexityOptions to be used for LLM-based complexity evaluation.
This must be called before running equation_search if you want to use LLM complexity.
"""
function set_active_llm_options(options::ComplexityOptions)
    ACTIVE_LLM_OPTIONS[] = options
    return nothing
end

"""
    clear_active_llm_options()

Clear the active LLM options, reverting to standard complexity computation.
"""
function clear_active_llm_options()
    ACTIVE_LLM_OPTIONS[] = nothing
    return nothing
end

"""
    _original_compute_complexity_impl(tree, options; break_sharing)

Reimplementation of the original compute_complexity logic from SymbolicRegression.jl.
This avoids recursion by directly implementing the same logic.
"""
function _original_compute_complexity_impl(
    tree::AbstractExpression,
    options::AbstractOptions;
    break_sharing=Val(false)
)
    if options.complexity_mapping isa Function
        return options.complexity_mapping(tree)::Int
    else
        return _original_compute_complexity_impl(get_tree(tree), options; break_sharing)
    end
end

function _original_compute_complexity_impl(
    tree::AbstractExpressionNode,
    options::AbstractOptions;
    break_sharing=Val(false)
)::Int
    complexity_mapping = options.complexity_mapping # Apply the complexity mapping if the user provides it
    if complexity_mapping isa ComplexityMapping && complexity_mapping.use
        # Call the internal _compute_complexity from SymbolicRegression
        raw_complexity = SR.ComplexityModule._compute_complexity(
            tree, options.complexity_mapping; break_sharing
        )
        return round(Int, raw_complexity)
    else
        return count_nodes(tree; break_sharing)
    end
end

"""
    _patched_compute_complexity(tree, options; break_sharing)

The patched implementation of compute_complexity that will be installed at runtime.
This intercepts all calls to compute_complexity within SymbolicRegression.jl.
"""
function _patched_compute_complexity(
    tree::Union{AbstractExpression, AbstractExpressionNode},
    options::AbstractOptions;
    break_sharing=Val(false)
)
    # Check if LLM complexity is active
    active_opts = ACTIVE_LLM_OPTIONS[]

    if active_opts !== nothing && active_opts.use_llm_complexity
        # Use LLM-based complexity evaluation
        llm_func = LLM_COMPLEXITY_FUNC[]
        if llm_func !== nothing
            return llm_func(tree, active_opts)
        else
            error("LLM complexity function not set. This is a bug in SR_with_LLM_Complexity.")
        end
    else
        # Fall back to original SymbolicRegression complexity implementation
        complexity = _original_compute_complexity_impl(tree, options; break_sharing=break_sharing)

        # If we have active options with logging enabled, log the standard complexity
        if active_opts !== nothing && active_opts.log_complexity_outputs && !active_opts.use_llm_complexity
            log_func = STANDARD_COMPLEXITY_LOG_FUNC[]
            if log_func !== nothing
                log_func(complexity, active_opts.log_standard_file_path)
            end
        end

        return complexity
    end
end

"""
    set_llm_complexity_function(f::Function)

Set the function to use for LLM complexity computation.
This is called by the main module after LLMComplexity is loaded.
"""
function set_llm_complexity_function(f::Function)
    LLM_COMPLEXITY_FUNC[] = f
    return nothing
end

"""
    set_standard_complexity_log_function(f::Function)

Set the function to use for logging standard complexity.
This is called by the main module after LLMComplexity is loaded.
"""
function set_standard_complexity_log_function(f::Function)
    STANDARD_COMPLEXITY_LOG_FUNC[] = f
    return nothing
end

"""
    install_patches!()

Install all method patches to SymbolicRegression.jl functions.
This must be called at runtime (not during precompilation), typically from __init__().
"""
function install_patches!()
    # Capture patched function reference to interpolate into @eval
    _patched = _patched_compute_complexity

    # Patch compute_complexity in SymbolicRegression's ComplexityModule
    # The patched function calls _compute_complexity internally to avoid recursion
    @eval SR.ComplexityModule begin
        function compute_complexity(
            tree::$AbstractExpression,
            options::$AbstractOptions;
            break_sharing=Val(false)
        )
            return $_patched(tree, options; break_sharing=break_sharing)
        end
    end

    return nothing
end

end # module ComplexityPatchingModule
