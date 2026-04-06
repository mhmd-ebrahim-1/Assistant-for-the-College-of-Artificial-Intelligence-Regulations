"""Compatibility entrypoint.

Deprecated: use `python app/main.py`.
"""

import warnings

from app.main import app

warnings.warn(
    "`flask_app.py` is deprecated. Use `python app/main.py`.",
    DeprecationWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
