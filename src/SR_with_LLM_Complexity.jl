module SR_with_LLM_Complexity

using Reexport

# Re-export everything from SymbolicRegression so users can use this as a drop-in replacement
@reexport using SymbolicRegression

# Import the specific function we want to override
import SymbolicRegression.ComplexityModule: compute_complexity

# Include modules in dependency order (order matters!)
# 1. Tools module (no dependencies)
include("Tools.jl")
using .Tools

# 2. Options struct (no dependencies)
include("LLMComplexityOptionsStruct.jl")
using .LLMComplexityOptionsStructModule

# 3. LLMComplexity module (depends on Tools)
include("LLMComplexity.jl")
using .LLMComplexity: compute_complexity  # Use our overridden version

# 4. Options constructor (depends on LLMComplexity for initialize_log)
include("LLMComplexityOptions.jl")
using .LLMComplexityOptionsModule: ComplexityOptions

# Export additional functionality
export string_tree_llm
export LLMComplexityOptions, ComplexityOptions

end # module SR_with_LLM_Complexity
