from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from controls import (  # noqa: E402
    action_id,
    camera_poses,
    expand_action_plan,
    latent_frames_for,
)


class ControlTests(unittest.TestCase):
    def test_geometry(self) -> None:
        self.assertEqual(latent_frames_for(121), 16)
        self.assertEqual(latent_frames_for(729), 92)
        with self.assertRaises(ValueError):
            latent_frames_for(129)

    def test_action_plan_exact_and_implicit_tail(self) -> None:
        exact = expand_action_plan("W:5,right:5,stop:5", transition_count=15)
        self.assertEqual(exact, ["W"] * 5 + ["right"] * 5 + ["stop"] * 5)
        implicit = expand_action_plan("W+D:8,left", transition_count=15)
        self.assertEqual(implicit, ["W+D"] * 8 + ["left"] * 7)

    def test_invalid_duration_total(self) -> None:
        with self.assertRaises(ValueError):
            expand_action_plan("W:5,right:5", transition_count=15)

    def test_action_ids(self) -> None:
        self.assertEqual(action_id("stop"), 0)
        self.assertEqual(action_id("W"), 9)
        self.assertEqual(action_id("right"), 1)
        self.assertEqual(action_id("W+D"), 45)

    def test_pose_count(self) -> None:
        actions = ["W"] * 15
        self.assertEqual(len(camera_poses(actions, perspective="first_person")), 16)
        self.assertEqual(len(camera_poses(actions, perspective="third_person")), 16)


if __name__ == "__main__":
    unittest.main()
