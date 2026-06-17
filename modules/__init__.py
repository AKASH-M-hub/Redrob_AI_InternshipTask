"""`modules` package init.

This file intentionally avoids importing submodules at package import time to
prevent circular imports (importing `modules` should be lightweight).

Use explicit imports like `from modules import pipeline` or
`from modules.pipeline import analyze_document` where needed.
"""

__all__ = []