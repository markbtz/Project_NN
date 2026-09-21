import csv
import pytest

from scripts.compare_evaluations import (
    align_by_image_id,
    compute_differences,
    load_per_image_csv,
)


def write_csv(path, rows):
    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "cc",
                "sim",
                "kld",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def test_alignment_uses_image_id_not_row_order(
    tmp_path,
):
    left_path = tmp_path / "b0.csv"
    right_path = tmp_path / "b1.csv"

    write_csv(
        left_path,
        [
            {
                "image_id": "img_1",
                "cc": 0.50,
                "sim": 0.60,
                "kld": 0.70,
            },
            {
                "image_id": "img_2",
                "cc": 0.40,
                "sim": 0.50,
                "kld": 0.80,
            },
        ],
    )

    write_csv(
        right_path,
        [
            {
                "image_id": "img_2",
                "cc": 0.70,
                "sim": 0.80,
                "kld": 0.30,
            },
            {
                "image_id": "img_1",
                "cc": 0.80,
                "sim": 0.90,
                "kld": 0.20,
            },
        ],
    )

    left_samples = load_per_image_csv(left_path)
    right_samples = load_per_image_csv(right_path)

    aligned = align_by_image_id(
        left_samples,
        right_samples,
        left_name="B0",
        right_name="B1",
    )

    assert aligned[0][0] == "img_1"
    assert aligned[0][1]["cc"] == 0.50
    assert aligned[0][2]["cc"] == 0.80

    assert aligned[1][0] == "img_2"
    assert aligned[1][1]["cc"] == 0.40
    assert aligned[1][2]["cc"] == 0.70

def test_alignment_rejects_missing_image_ids(
    tmp_path,
):
    left_path = tmp_path / "b0.csv"
    right_path = tmp_path / "b1.csv"

    write_csv(
        left_path,
        [
            {
                "image_id": "img_1",
                "cc": 0.50,
                "sim": 0.60,
                "kld": 0.70,
            },
            {
                "image_id": "img_2",
                "cc": 0.40,
                "sim": 0.50,
                "kld": 0.80,
            },
        ],
    )

    write_csv(
        right_path,
        [
            {
                "image_id": "img_1",
                "cc": 0.80,
                "sim": 0.90,
                "kld": 0.20,
            },
        ],
    )

    left_samples = load_per_image_csv(left_path)
    right_samples = load_per_image_csv(right_path)

    with pytest.raises(
        ValueError,
        match="non contengono gli stessi image_id",
    ):
        align_by_image_id(
            left_samples,
            right_samples,
            left_name="B0",
            right_name="B1",
        )

def test_load_per_image_csv_rejects_duplicate_ids(
    tmp_path,
):
    path = tmp_path / "duplicates.csv"

    write_csv(
        path,
        [
            {
                "image_id": "img_1",
                "cc": 0.50,
                "sim": 0.60,
                "kld": 0.70,
            },
            {
                "image_id": "img_1",
                "cc": 0.80,
                "sim": 0.90,
                "kld": 0.20,
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="image_id duplicato",
    ):
        load_per_image_csv(path)

def test_compute_differences_uses_right_minus_left():
    aligned = [
        (
            "img_1",
            {
                "cc": 0.50,
                "sim": 0.60,
                "kld": 0.70,
            },
            {
                "cc": 0.80,
                "sim": 0.90,
                "kld": 0.20,
            },
        ),
    ]

    rows = compute_differences(
        aligned,
        left_name="B0",
        right_name="B1",
    )

    assert len(rows) == 1

    row = rows[0]

    assert row["image_id"] == "img_1"

    assert row["delta_cc"] == pytest.approx(0.30)
    assert row["delta_sim"] == pytest.approx(0.30)
    assert row["delta_kld"] == pytest.approx(-0.50)