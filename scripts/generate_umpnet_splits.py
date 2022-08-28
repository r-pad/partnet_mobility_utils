import json
import logging
from pathlib import Path

import typer

from partnet_mobility_utils.dataset import read_ids, write_ids


def main(split_path: Path, out_dir: Path = "partnet_mobility_utils/data_lists/umpnet"):
    with split_path.open("r") as f:
        contents = json.load(f)

    all_ids = read_ids(
        Path(__file__).parent.parent
        / "partnet_mobility_utils"
        / "data_lists"
        / "all_ids.txt"
    )
    all_cats = {cat for _, cat in all_ids}

    def extract_ids(obj_split, inst_split):
        obj_ids = []
        obj_dict = contents[obj_split]
        for cat, inst_dict in obj_dict.items():
            if cat not in all_cats:
                logging.warning(
                    f"{obj_split}:{inst_split} Skipping category: {cat}, not a PM category"
                )
                continue
            inst_ids = inst_dict[inst_split]
            obj_ids.extend([(id, cat) for id in inst_ids])
        return obj_ids

    train_train_ids = extract_ids("train", "train")
    train_test_ids = extract_ids("train", "test")
    test_ids = extract_ids("test", "test")

    write_ids(out_dir / "train_train.txt", sorted(train_train_ids))
    write_ids(out_dir / "train_test.txt", sorted(train_test_ids))
    write_ids(out_dir / "test.txt", sorted(test_ids))


if __name__ == "__main__":
    typer.run(main)
