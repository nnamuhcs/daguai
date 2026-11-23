"""Render the repository README as the GitHub Pages index."""

import argparse
from html import escape
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/nnamuhcs/daguai"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css">
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; background: #f3f6f7; color: #172730; font: 1rem/1.65 system-ui, sans-serif; }}
  main {{ max-width: 62rem; margin: auto; padding: clamp(1rem, 5vw, 3rem); background: white; }}
  h1, h2, h3 {{ line-height: 1.25; }}
  h1 {{ font-size: clamp(1.7rem, 4vw, 2.5rem); }}
  a {{ color: #075d79; overflow-wrap: anywhere; }}
  pre, blockquote, table {{ background: #f3f6f7; }}
  pre, blockquote {{ margin-inline: 0; padding: .8rem 1rem; }}
  pre, .math.block, .mermaid {{ overflow-x: auto; }}
  blockquote {{ border-left: .25rem solid #438c9e; }}
  table {{ display: block; overflow-x: auto; border-collapse: collapse; }}
  th, td {{ border: 1px solid #d5e0e4; padding: .35rem .7rem; }}
  details > summary {{ cursor: pointer; }}
</style>
</head>
<body>
<main>
<nav aria-label="Project links"><a href="{repository}">View source on GitHub</a></nav>
{article}
</main>
<script type="module">
  import katex from 'https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.mjs';
  document.querySelectorAll('.math').forEach(element => {{
    katex.render(element.textContent, element, {{
      displayMode: element.classList.contains('block'), throwOnError: false
    }});
  }});
</script>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11.12.0/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{startOnLoad: false, securityLevel: 'strict'}});
  await mermaid.run({{querySelector: '.mermaid'}});
</script>
</body>
</html>
"""


def build(output):
    parser = (MarkdownIt("commonmark", {"html": True})
              .enable("table").use(anchors_plugin, max_level=6).use(dollarmath_plugin))

    def render_fence(tokens, index, options, env):
        token = tokens[index]
        if token.info.strip() == "mermaid":
            return f'<pre class="mermaid">{escape(token.content)}</pre>\n'
        return parser.renderer.fence(tokens, index, options, env)

    parser.renderer.rules["fence"] = render_fence
    env = {}
    tokens = parser.parse((ROOT / "README.md").read_text(encoding="utf-8"), env)
    title = next(inline.content for heading, inline in zip(tokens, tokens[1:]) if heading.tag == "h1")
    for token in tokens:
        for child in token.children or ():
            if child.type == "link_open":
                href = child.attrGet("href")
                if href and not href.startswith("#") and not urlsplit(href).scheme:
                    child.attrSet("href", urljoin(f"{REPOSITORY}/blob/main/", href))
    article = parser.renderer.render(tokens, parser.options, env)
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(
        PAGE.format(title=escape(title), repository=REPOSITORY, article=article), encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    build(cli.parse_args().output)
