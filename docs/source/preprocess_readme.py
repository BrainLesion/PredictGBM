import re


def preprocess_readme(input_file: str, output_file: str) -> None:
    """Convert GitHub-style admonitions to MyST/Sphinx syntax.

    Args:
        input_file (str): original README file
        output_file (str): processed README file

    """
    with open(input_file, "r") as file:
        content = file.read()

    admonition_types = ["IMPORTANT", "NOTE", "TIP", "WARNING", "CAUTION"]

    for ad_type in admonition_types:
        # Replace > [!ad_type] with MyST admonition syntax
        content = re.sub(
            r"> \[!"
            + ad_type
            + r"\]\s*\n((?:> .*\n)*)",  # Match the > [!ad_type] and subsequent lines
            lambda m: "```{"
            + ad_type
            + "}\n"
            + m.group(1).replace("> ", "").strip()
            + "\n```",
            content,
        )

    # Replace empty Markdown links with plain text to avoid Sphinx warnings
    content = re.sub(r"\[([^\]]+)\]\(\s*\)", r"\1", content)

    # Convert relative repository links to absolute GitHub URLs
    repo_base = "https://github.com/BrainLesion/PredictGBM/blob/main/"
    content = re.sub(
        r"\[([^\]]+)\]\((scripts/[^)]+)\)",
        lambda m: f"[{m.group(1)}]({repo_base}{m.group(2)})",
        content,
    )
    content = re.sub(
        r"\[([^\]]+)\]\((CONTRIBUTING\.md)\)",
        lambda m: f"[{m.group(1)}]({repo_base}{m.group(2)})",
        content,
    )

    # Write the transformed content to the output file
    with open(output_file, "w") as file:
        file.write(content)


if __name__ == "__main__":
    preprocess_readme("../../README.md", "../README_preprocessed.md")
