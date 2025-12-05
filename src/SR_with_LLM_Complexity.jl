module SR_with_LLM_Complexity

using Reexport

# Re-export everything from SymbolicRegression so users can use this as a drop-in replacement
@reexport using SymbolicRegression

# Include modules in dependency order (order matters!)
# 1. Tools module (no dependencies)
include("Tools.jl")
using .Tools

# 2. Options struct (no dependencies on other modules)
include("LLMComplexityOptionsStruct.jl")
using .LLMComplexityOptionsStructModule

# 3. Patching module (depends on Options struct)
include("ComplexityPatching.jl")
using .ComplexityPatchingModule

# 4. LLMComplexity module (depends on Tools and Options)
include("LLMComplexity.jl")
using .LLMComplexity: compute_llm_complexity, log_complexity

# 5. Options constructor (depends on LLMComplexity for initialize_log)
include("LLMComplexityOptions.jl")
using .LLMComplexityOptionsModule: ComplexityOptions

# 6. Equation search wrapper (depends on ComplexityOptions and Patching module)
include("EquationSearchWrapper.jl")
using .EquationSearchWrapperModule: equation_search

# Export additional functionality
export string_tree_llm
export LLMComplexityOptions, ComplexityOptions
export set_active_llm_options, clear_active_llm_options
export equation_search

# Runtime initialization - install method patches after precompilation
function __init__()
    # Set the LLM complexity function in the patching module
    ComplexityPatchingModule.set_llm_complexity_function(compute_llm_complexity)

    # Set the standard complexity logging function
    ComplexityPatchingModule.set_standard_complexity_log_function(log_complexity)

    # Install patches to SymbolicRegression
    ComplexityPatchingModule.install_patches!()

    println("SR_with_LLM_Complexity: Runtime patches installed. Precompilation enabled!")
end

end # module SR_with_LLM_Complexity
