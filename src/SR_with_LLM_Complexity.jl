module SR_with_LLM_Complexity

using Reexport

# Re-export everything from SymbolicRegression so users can use this as a drop-in replacement
@reexport using SymbolicRegression

# Import the specific function we want to override
import SymbolicRegression.ComplexityModule: compute_complexity

# Include our custom complexity module
include("LLMComplexity.jl")
using .LLMComplexity: compute_complexity  # Use our overridden version

# Include other modules
include("Tools.jl")
using .Tools

# Export additional functionality if needed
export string_tree_llm

end # module SR_with_LLM_Complexity
