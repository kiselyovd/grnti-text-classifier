"""Patch pyproject.toml to remove [tool.uv.sources] and [[tool.uv.index]] sections.

These sections pin torch to the CUDA index. We remove them so uv resolves
torch from whatever index we pass on the CLI (the CPU wheel index).
"""
import pathlib
import re

p = pathlib.Path("pyproject.toml")
txt = p.read_text()

# Disable [tool.uv.sources] section by renaming its header
txt = re.sub(r"\[tool\.uv\.sources\]", "[tool._uv_sources_disabled]", txt)

# Disable [[tool.uv.index]] sections by renaming their headers
txt = re.sub(r"\[\[tool\.uv\.index\]\]", "[[tool._uv_index_disabled]]", txt)

p.write_text(txt)
print("patched pyproject.toml: disabled [tool.uv.sources] and [[tool.uv.index]]")
