"""
EquationSearchWrapper.jl

Wrapper for equation_search that automatically manages LLM complexity options.
"""
module EquationSearchWrapperModule

using SymbolicRegression: equation_search as sr_equation_search
using ..LLMComplexityOptionsStructModule: ComplexityOptions
using ..ComplexityPatchingModule: set_active_llm_options, clear_active_llm_options

export equation_search

"""
    equation_search(X, y; niterations, options::ComplexityOptions, kwargs...)

Wrapper around SymbolicRegression's equation_search that automatically manages
complexity options via the patching system.

This wrapper:
1. Sets the active ComplexityOptions before the search (for both LLM and standard complexity)
2. Calls the original equation_search
3. Clears the active options after the search

This ensures that the patched compute_complexity function has access to the
ComplexityOptions during the search (for LLM complexity computation or standard complexity logging).
"""
function equation_search(X, y; niterations::Int, options::ComplexityOptions, kwargs...)
    # Always set active options (needed for both LLM complexity and standard complexity logging)
    set_active_llm_options(options)

    try
        # Call the original equation_search from SymbolicRegression
        # The patched compute_complexity will use our LLM function or log standard complexity
        result = sr_equation_search(
            X, y;
            niterations=niterations,
            options=options.sr_options,  # Pass the wrapped SR Options
            kwargs...
        )
        return result
    finally
        # Always clear active options after search, even if error occurs
        clear_active_llm_options()
    end
end

end # module EquationSearchWrapperModule
