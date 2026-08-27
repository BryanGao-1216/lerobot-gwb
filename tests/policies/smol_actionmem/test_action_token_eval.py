from types import SimpleNamespace

import torch
from torch.utils.data import default_collate

from action_token_eval.evaluate_action_tokens import _collate_single_sample, _write_interactive_html


def test_collate_single_eval_sample_preserves_image_mask_batch_dimension():
    camera_key = "observation.images.image"
    dataset = SimpleNamespace(
        collate_fn=default_collate,
        meta=SimpleNamespace(camera_keys=[camera_key]),
    )
    sample = {
        camera_key: torch.zeros(3, 8, 8, dtype=torch.uint8),
        f"{camera_key}_padding_mask": torch.tensor(True),
    }

    batch = _collate_single_sample(dataset, sample)

    assert batch[camera_key].shape == (1, 3, 8, 8)
    assert batch[f"{camera_key}_padding_mask"].shape == (1,)


def test_interactive_report_is_one_self_contained_html_without_raster_images(tmp_path):
    output_path = tmp_path / "action_token_eval.html"
    report = {
        "seed": 42,
        "top_k": 1,
        "samples": [
            {
                "sample": 1,
                "dataset_name": "demo",
                "task": "move </script> safely",
                "top_tokens": torch.tensor([7]),
                "top_probabilities": torch.tensor([0.75]),
                "decoded_effect_prototypes": torch.zeros(1, 7),
                "true_action_chunk": torch.zeros(2, 7),
            }
        ],
    }

    _write_interactive_html(output_path, report)

    html = output_path.read_text(encoding="utf-8")
    assert "<canvas id=\"position-chart\"" in html
    assert "<canvas id=\"rotation-chart\"" in html
    assert "<canvas id=\"gripper-chart\"" in html
    assert "move <\\/script> safely" in html
    assert "__ACTION_TOKEN_EVAL_DATA__" not in html
    assert "<img" not in html
    assert not list(tmp_path.glob("*.png"))
