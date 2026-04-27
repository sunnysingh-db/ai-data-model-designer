#!/usr/bin/env python3
"""Convert assets/logo.png to assets/logo_data.js (base64-encoded for PDF export).

Usage:
    python scripts/generate_logo_b64.py

Run this after replacing assets/logo.png with your own logo.
"""

import base64
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    logo_path = os.path.join(project_dir, "assets", "logo.png")
    output_path = os.path.join(project_dir, "assets", "logo_data.js")

    if not os.path.exists(logo_path):
        print(f"Error: {logo_path} not found.")
        print("Place your logo as assets/logo.png and rerun.")
        return

    with open(logo_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    js_content = (
        '// Auto-generated from assets/logo.png\n'
        '// Regenerate by running: python scripts/generate_logo_b64.py\n'
        f'window.__LOGO_B64__ = "data:image/png;base64,{b64}";\n'
    )

    with open(output_path, "w") as f:
        f.write(js_content)

    print(f"Generated {output_path} ({len(b64)} chars base64)")


if __name__ == "__main__":
    main()
