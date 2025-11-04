"""
start_llm_server.jl

Starts the LLM server on localhost:11449
Run this before executing tests that require LLM complexity evaluation.

Usage:
    julia --project=. src/start_llm_server.jl
"""

include("LLMServe.jl")
using .LLMServeModule

println("=" ^ 60)
println("Starting LLM Server")
println("=" ^ 60)
println("Model: ", LLMServeModule.LLAMAFILE_MODEL)
println("Path: ", LLMServeModule.LLAMAFILE_PATH)
println("Port: ", LLMServeModule.LLM_PORT)
println("URL: http://localhost:", LLMServeModule.LLM_PORT)
println("=" ^ 60)

# Start the server
proc = LLMServeModule.async_run_llm_server()

println("\n✓ LLM Server started (PID: ", getpid(proc), ")")
println("✓ Server will run in the background")
println("✓ Server will automatically shut down when Julia exits")
println("\nPress Ctrl+C to stop or just close this terminal")
println("Run 'kill ", getpid(proc), "' to stop manually\n")

# Keep the script running so the server stays alive
try
    wait(proc)
catch e
    if isa(e, InterruptException)
        println("\n\n✓ Shutting down LLM server...")
        kill(proc, Base.SIGTERM)
        wait(proc)
        println("✓ Server stopped")
    else
        rethrow(e)
    end
end
