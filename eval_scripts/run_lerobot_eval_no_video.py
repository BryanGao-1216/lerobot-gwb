#!/usr/bin/env python
"""Drop-in LeRobot entrypoint: disables video creation without changing rollout or policy code."""
# Import Wand before other libraries to avoid a potential ImageMagick dlopen-order conflict.
from wand.api import library as _magickwand  # noqa: F401

import lerobot.scripts.lerobot_eval as evaluator

_original_eval_policy_all = evaluator.eval_policy_all


def _eval_policy_all_without_video(*args, **kwargs):
    # The upstream CLI's --eval.recording=false still saves up to ten videos per task.
    # Turning off rendering only removes the extra per-step render callback.
    kwargs["max_episodes_rendered"] = 0
    kwargs["videos_dir"] = None
    return _original_eval_policy_all(*args, **kwargs)


evaluator.eval_policy_all = _eval_policy_all_without_video

if __name__ == "__main__":
    evaluator.main()
