"""QuantiPhy 官方验证 CSV 的只读入口；题目输入与参考答案分开读取。"""

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class QuantiPhyTask:
    question_id: str
    video_id: str
    video_type: str
    inference_type: str
    fps: float
    question: str
    prior: str
    depth_info: str

    @property
    def category(self) -> str:
        """对齐官方 evaluator 的 S2、D2、S3、D3；不是 SS/SD/DS/DD。"""
        return self.inference_type[0] + self.video_type[1]


def _rows(path: str | Path) -> tuple[str, list[dict[str, str]]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        headers = next(reader, None)
        if not headers:
            raise ValueError("dataset CSV must have a header")
        # 官方 CSV 首列 ID 无标题，末尾还带多个空标题列。
        # DictReader 会将同名空标题覆盖，必须按列位置保留首列 ID。
        id_column = "_official_id"
        rows = []
        for cells in reader:
            if not cells or not any(cell.strip() for cell in cells):
                continue
            row = {name: cells[index] if index < len(cells) else ""
                   for index, name in enumerate(headers) if index > 0 and name.strip()}
            row[id_column] = cells[0]
            rows.append(row)
    ids = [row[id_column] for row in rows]
    if any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("question IDs must be nonempty and unique")
    return id_column, rows


def load_tasks(path: str | Path) -> list[QuantiPhyTask]:
    """读取输入，保留官方 ID；不把 ground_truth_posterior 放入任务对象。

    ground_truth_prior 是题目给定的已知量，可以作为输入，区别于目标答案。
    仅加载问题表，不下载视频、不解析自然语言或声称已完成物体跟踪。
    """
    id_column, rows = _rows(path)
    tasks = []
    for row in rows:
        video_type = row["video_type"]
        inference_type = row["inference_type"]
        fps = float(row["fps"])
        if len(video_type) < 2 or video_type[1] not in "23" or inference_type not in {"SS", "SD", "DS", "DD"}:
            raise ValueError("unknown QuantiPhy category")
        if not isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        tasks.append(QuantiPhyTask(
            question_id=row[id_column], video_id=row["video_id"],
            video_type=video_type, inference_type=inference_type, fps=fps,
            question=row["question"], prior=row["ground_truth_prior"],
            depth_info=row.get("depth_info", ""),
        ))
    return tasks


def load_validation_answers(path: str | Path) -> dict[str, float]:
    """仅供本地评测读取参考答案；禁止传入轨迹估计与模型输入。"""
    id_column, rows = _rows(path)
    answers = {row[id_column]: float(row["ground_truth_posterior"]) for row in rows}
    if not all(isfinite(value) for value in answers.values()):
        raise ValueError("validation answers must be finite")
    return answers
