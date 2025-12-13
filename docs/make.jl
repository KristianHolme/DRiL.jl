using Documenter
using DocumenterVitepress
using DRiL

makedocs(;
    sitename = "DRiL.jl",
    authors = "Kristian Holme",
    modules = [DRiL],
    warnonly = true,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "https://github.com/KristianHolme/DRiL.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Guide" => [
            "Environments" => "guide/environments.md",
            "Algorithms" => "guide/algorithms.md",
            "Wrappers" => "guide/wrappers.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/KristianHolme/DRiL.jl",
    target = "build",
    push_preview = true,
    devbranch = "main",
)
