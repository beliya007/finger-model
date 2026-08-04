import json
import tempfile
from pathlib import Path

from ivsf_pipeline.io_utils import save_timing_summary


def test_save_timing_summary_writes_json_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "timing_stats.json"
        payload = {
            "candidate_selection_seconds": 0.1,
            "optimization_seconds_per_candidate_base": [0.2],
            "generation_seconds_per_candidate_base": [0.3],
        }

        save_timing_summary(path, payload)

        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["candidate_selection_seconds"] == 0.1
        assert loaded["optimization_seconds_per_candidate_base"] == [0.2]
