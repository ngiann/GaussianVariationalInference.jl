using Documenter, ApproximateVI

makedocs(
    sitename = "ApproximateVI",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    clean = true,
    authors = "Nikos Gianniotis",
    pages = ["Introduction"      => "index.md", 
             "More options"      => "moreoptions.md",
             "Technical description" => "technicaldescription.md",
             "Examples"          => "examples.md",
             "Reference"         => "reference.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/ngiann/ApproximateVI.jl",
)
