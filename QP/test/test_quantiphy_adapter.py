import csv
from dataclasses import asdict

import pytest

from adapters.quantiphy import load_tasks, load_validation_answers


def write_sample(path, answer="3.7", duplicate=False):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["", "video_id", "video_type", "inference_type", "fps", "question", "ground_truth_prior", "depth_info", "ground_truth_posterior", "", ""])
        row = ["2161", "internet_0019", "A2MC", "DD", "30", "What is the speed in m/s?", "gravity acc = 9.8 m/s^2", "", answer, "", "#NAME?"]
        writer.writerow(row)
        if duplicate:
            writer.writerow(row)


def test_official_id_prior_and_category_are_preserved(tmp_path):
    path = tmp_path / "sample.csv"
    write_sample(path)
    task = load_tasks(path)[0]
    assert task.question_id == "2161"
    assert task.prior == "gravity acc = 9.8 m/s^2"
    assert task.category == "D2"
    assert task.fps == 30.0
    assert load_validation_answers(path) == {"2161": 3.7}


def test_target_answers_do_not_enter_task_inputs(tmp_path):
    path = tmp_path / "sample.csv"
    write_sample(path, "3.7")
    before = load_tasks(path)
    write_sample(path, "999999")
    assert load_tasks(path) == before
    assert "ground_truth_posterior" not in asdict(before[0])
    assert "answer" not in asdict(before[0])


def test_duplicate_official_ids_are_rejected(tmp_path):
    path = tmp_path / "sample.csv"
    write_sample(path, duplicate=True)
    with pytest.raises(ValueError, match="unique"):
        load_tasks(path)
