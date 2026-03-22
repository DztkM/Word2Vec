import re


class TextCleaner:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            text = f.read()

        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved cleaned text to: {self.output_path}")


if __name__ == "__main__":
    cleaner = TextCleaner(
        input_path="data/alice_raw.txt",
        output_path="data/alice_clean.txt"
    )
    cleaner.run()