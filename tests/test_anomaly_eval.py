import copy
import unittest

from weldfusion.anomaly_eval import auc, audio_mean, evaluate, video_peak_mean


def row(sample_id, split, category, audio, video):
    return {"sample_id": sample_id, "split": split, "category": category,
            "audio_score": audio, "video_score": video}


class EvaluationTests(unittest.TestCase):
    def test_audio_and_video_aggregation(self):
        self.assertEqual(audio_mean([1, 3, 5]), 3)
        self.assertEqual(video_peak_mean([0, 1, 5, 5, 0], 2), 5)
        with self.assertRaises(ValueError):
            video_peak_mean([1], 2)

    def test_ties_in_auc(self):
        self.assertEqual(auc([1, 1], [0, 1]), 0.5)
        self.assertEqual(auc([0, 1, 2, 3], [0, 0, 1, 1]), 1)

    def test_validation_weight_does_not_use_test_labels(self):
        records = [row("t0", "TRAIN", "Good", 0, 0),
                   row("t1", "TRAIN", "Good", 1, 1),
                   row("t2", "TRAIN", "Good", 2, 2),
                   row("v0", "VAL", "Good", 1, 1),
                   row("v1", "VAL", "Spatter", 10, -10),
                   row("v2", "VAL", "Porosity", 9, -9),
                   row("x0", "TEST", "Good", 1, 1),
                   row("x1", "TEST", "Spatter", -10, 10),
                   row("x2", "TEST", "Porosity", -9, 9)]
        result = evaluate(records)
        self.assertEqual(result["validation"]["auc"]["fusion"], 1)
        self.assertEqual(result["test"]["auc"]["fusion"], 0)
        self.assertEqual(set(result["test"]["per_defect_auc"]), {"Spatter", "Porosity"})
        changed = copy.deepcopy(records)
        changed[-1]["audio_score"] = 100
        changed[-1]["video_score"] = -100
        self.assertEqual(result["selected_audio_weight"], evaluate(changed)["selected_audio_weight"])

    def test_defects_cannot_enter_training_normalization(self):
        records = [row("t0", "TRAIN", "Good", 0, 0),
                   row("t1", "TRAIN", "Spatter", 1, 1),
                   row("v0", "VAL", "Good", 0, 0),
                   row("v1", "VAL", "Spatter", 1, 1),
                   row("x0", "TEST", "Good", 0, 0),
                   row("x1", "TEST", "Spatter", 1, 1)]
        with self.assertRaisesRegex(ValueError, "Good welds only"):
            evaluate(records)


if __name__ == "__main__":
    unittest.main()
