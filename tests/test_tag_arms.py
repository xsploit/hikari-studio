"""Reply mood tags leave arm poses to the rig and explicit waves."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.ai_companion import TAG_TO_STATE


class ReplyTagTests(unittest.TestCase):
    def test_wave_is_the_only_tag_with_arm_values(self) -> None:
        arm_fields = {"armR", "armL"}
        actual = {
            tag: arm_fields & state.keys()
            for tag, state in TAG_TO_STATE.items()
            if arm_fields & state.keys()
        }
        self.assertEqual(actual, {"wave": arm_fields})


if __name__ == "__main__":
    unittest.main()
