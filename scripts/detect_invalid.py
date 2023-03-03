from pathlib import Path

import typer

from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.dataset import write_ids


def main(root: Path, outdir: Path):
    assert root.is_dir()
    assert outdir.is_dir()

    well_formed = []
    extra_objs = []
    unworkable = []
    all_ids = []
    all_cats = set()
    for odir in sorted(root.iterdir()):
        obj = PMObject(odir)
        obj_id = obj.obj_id
        model_cat = obj.category
        issame, issubset = obj.well_formed, obj.usable

        all_cats.add(model_cat)

        all_ids.append((obj_id, model_cat))

        if not issubset:
            unworkable.append((obj_id, model_cat))

        if not issame and issubset:
            extra_objs.append((obj_id, model_cat))

        if issame:
            well_formed.append((obj_id, model_cat))

    all_ids_fn = outdir / "all_ids.txt"
    well_formed_fn = outdir / "well_formed.txt"
    extra_objs_fn = outdir / "extra_objs.txt"
    unworkable_fn = outdir / "missing_objs.txt"

    write_ids(all_ids_fn, sorted(all_ids))
    write_ids(well_formed_fn, sorted(well_formed))
    write_ids(extra_objs_fn, sorted(extra_objs))
    write_ids(unworkable_fn, sorted(unworkable))


if __name__ == "__main__":
    typer.run(main)
