module SR_with_LLM_Complexity

using Reexport

# Re-export everything from SymbolicRegression so users can use this as a drop-in replacement
@reexport using SymbolicRegression

# Import the specific function we want to override
import SymbolicRegression.ComplexityModule: compute_complexity

# Include options modules first (order matters!)
include("LLMComplexityOptionsStruct.jl")
using .LLMComplexityOptionsStructModule

include("LLMComplexityOptions.jl")
using .LLMComplexityOptionsModule

# Include other modules
include("Tools.jl")
using .Tools

# Include our custom complexity module
include("LLMComplexity.jl")
using .LLMComplexity: compute_complexity  # Use our overridden version

# Export additional functionality
export string_tree_llm
export LLMComplexityOptions, ComplexityOptions

end # module SR_with_LLM_Complexity
