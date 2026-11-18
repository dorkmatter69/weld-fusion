import io
import tarfile

import pytest

from weldfusion.archive import inventory


def _archive(path, names):
    with tarfile.open(path, "w:gz") as tar:
        for name, body in names:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))


def test_inventory_and_extract(tmp_path):
    archive = tmp_path / "small.tar.gz"
    _archive(archive, [("./data/a.csv", b"a,b\n1,2\n"), ("data/b.flac", b"fake")])
    report = inventory(archive, tmp_path / "staged")
    assert report["member_count"] == 2
    assert report["total_uncompressed_bytes"] == 12
    assert (tmp_path / "staged/data/a.csv").read_bytes() == b"a,b\n1,2\n"
    assert len(report["members"][0]["sha256"]) == 64


@pytest.mark.parametrize("name", ["../escape", "/absolute", "data/../escape", "data\\escape"])
def test_reject_unsafe_names(tmp_path, name):
    archive = tmp_path / "bad.tar.gz"
    _archive(archive, [(name, b"bad")])
    with pytest.raises(ValueError, match="unsafe"):
        inventory(archive, tmp_path / "staged")


def test_reject_symlink_parent(tmp_path):
    archive = tmp_path / "small.tar.gz"
    _archive(archive, [("data/a.csv", b"abc")])
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "data").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        inventory(archive, staged)
