from collections import defaultdict
from pathlib import Path

import typer

from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.dataset import write_ids


def main(root: Path, outdir: Path):
    assert root.is_dir()
    assert outdir.is_dir()

    all_cats = defaultdict(list)
    for odir in sorted(root.iterdir()):

        obj = PMObject(odir)
        obj_id = obj.obj_id
        model_cat = obj.category

        all_cats[model_cat].append((obj_id, model_cat))

    for cat, id_list in all_cats.items():
        outfile = outdir / f"{cat}.txt"
        write_ids(outfile, sorted(id_list))


if __name__ == "__main__":
    typer.run(main)
