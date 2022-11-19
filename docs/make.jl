using Documenter, ApproximateVI

makedocs(modules = [ApproximateVI],
format = Documenter.HTML(prettyurls = false),
checkdocs = :exports,
strict = true,
clean=true,
         sitename = "Approximate.jl",
         authors = "Nikos Gianniotis",
         pages = ["Introduction" => "index.md",
                    "Tecnical description" => "technicaldescription.md",
                    "More options" => "moreoptions.md",
                    "Documentation" => "interface.md"])

