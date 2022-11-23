using Documenter, ApproximateVI

makedocs(
    sitename = "ApproximateVI",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    clean = true,
    authors = "Nikos Gianniotis",
    pages = ["Introduction" => "index.md", "Reference" => "reference.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
