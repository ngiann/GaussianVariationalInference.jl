push!(LOAD_PATH,"../src/")

using Documenter, ApproximateVI

makedocs(modules = [ApproximateVI],
        format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
        checkdocs = :exports,
        strict = true,
        clean=true,
         sitename = "Approximate.jl",
         authors = "Nikos Gianniotis",
         pages = ["Introduction" => "index.md",
                    "More options" => "moreoptions.md",
                    "Technical description" => "technicaldescription.md",
                    "Documentation" => "interface.md"])

deploydocs(
    repo = "https://github.com/ngiann/ApproximateVI.jl",
)

