from pathlib import Path

from file_io.api import configure_logging, load_input, save_output


def main() -> None:
    configure_logging()

    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "input" / "tasks.json"
    output_dir = project_root / "output"

    tasks = load_input(input_path)
    output_path = save_output(tasks, output_dir)
    print(f"Wrote {len(tasks)} tasks to {output_path}")


if __name__ == "__main__":
    main()
