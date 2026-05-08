from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_agent_baseline.config import load_app_config_from_env
from data_agent_baseline.run.runner import run_benchmark

Path("/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/logs/main.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Data Agent...")

    config = load_app_config_from_env()
    if not config.agent.api_base or not config.agent.api_key:
        logger.error("Missing MODEL_API_URL or MODEL_API_KEY environment variables")
        sys.exit(1)

    logger.info(f"Model: {config.agent.model}")
    logger.info(f"API Base: {config.agent.api_base}")
    logger.info(f"Dataset root: {config.dataset.root_path}")
    logger.info(f"Output dir: {config.run.output_dir}")

    Path("/output").mkdir(parents=True, exist_ok=True)

    _, artifacts = run_benchmark(config=config)

    succeeded = sum(1 for a in artifacts if a.succeeded)
    logger.info(f"Done. {succeeded}/{len(artifacts)} tasks succeeded.")


if __name__ == "__main__":
    main()
