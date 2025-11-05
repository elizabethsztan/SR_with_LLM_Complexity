"""
LLMComplexityOptionsStruct.jl

Defines the options structures for LLM-based complexity evaluation.
"""

module LLMComplexityOptionsStructModule

using SymbolicRegression: AbstractOptions, Options

export LLMComplexityOptions, ComplexityOptions, LLM_COMPLEXITY_OPTIONS_KEYS

Base.@kwdef mutable struct LLMComplexityOptions
    # LLM Evaluation Control
    use_llm_complexity::Bool = false
    user_examples::String = "x1 + x2 + C has complexity 3, C * sin(x1) has complexity 4, sin(sin(sin(x1))) has complexity 10"
end

const LLM_COMPLEXITY_OPTIONS_KEYS = fieldnames(LLMComplexityOptions)

struct ComplexityOptions{O<:Options} <: AbstractOptions
    llm_complexity_options::LLMComplexityOptions
    sr_options::O
end

function Base.getproperty(options::ComplexityOptions, k::Symbol)
    if k === :llm_complexity_options
        return getfield(options, :llm_complexity_options)
    elseif k === :sr_options
        return getfield(options, :sr_options)
    elseif k in LLM_COMPLEXITY_OPTIONS_KEYS
        return getproperty(getfield(options, :llm_complexity_options), k)
    else
        return getproperty(getfield(options, :sr_options), k)
    end
end

function Base.propertynames(options::ComplexityOptions)
    return (LLM_COMPLEXITY_OPTIONS_KEYS..., fieldnames(Options)...)
end

end # module
