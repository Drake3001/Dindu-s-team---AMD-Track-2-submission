import json
from pathlib import Path
import structlog

from file_io.api import configure_logging, load_input, save_output
from file_io.download import download_for_task
from pipeline_2.grid_extractor.extractor import extract_smart_grids
from model_client.api import generate_from_images_base64
from model_client.prompts.loader import load_prompt

log = structlog.get_logger(__name__)

def main() -> None:
    configure_logging()

    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "input" / "tasks.json"
    output_dir = project_root / "output"
    videos_dir = project_root / "videos"

    tasks = load_input(input_path)
    results = []

    for task in tasks:
        task_id = task["task_id"]
        video_url = task["video_url"]
        styles = task.get("styles", [])
        
        log.info("processing_task", task_id=task_id)

        # 1. Download Video
        video_path = download_for_task(task_id, video_url, videos_dir=videos_dir)

        # 2. Extract Grids using Pipeline 2
        extraction_res = extract_smart_grids(video_path)
        grids_b64 = extraction_res["grids_b64"]
        
        task_result = {
            "task_id": task_id,
            "video_url": video_url,
            "generations": []
        }

        # 3. Prompt VLM for each style
        for style in styles:
            log.info("generating_style", task_id=task_id, style=style)
            prompt = load_prompt(style)
            
            response = generate_from_images_base64(
                images_base64=grids_b64,
                system_prompt=prompt.system,
                user_prompt=prompt.user
            )
            
            task_result["generations"].append({
                "style": style,
                "response": response
            })
            
        results.append(task_result)

    output_path = save_output(results, output_dir)
    print(f"Wrote {len(results)} task results to {output_path}")

if __name__ == "__main__":
    main()
