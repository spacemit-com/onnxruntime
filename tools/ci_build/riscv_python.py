# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Read target CPython artifacts without executing a RISC-V interpreter."""

import ast
import re
from email.parser import Parser
from pathlib import Path


def load_target_python(root):
    root = Path(root).resolve()
    configs = list(root.glob("lib/python3.*/_sysconfigdata_*.py"))
    if len(configs) != 1:
        raise ValueError(f"Expected one target Python _sysconfigdata file under {root}/lib/python3.*")
    variables = None
    for node in ast.parse(configs[0].read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "build_time_vars" for target in node.targets
        ):
            variables = ast.literal_eval(node.value)
            break
    if not isinstance(variables, dict):
        raise ValueError(f"Missing build_time_vars in {configs[0]}")
    version = variables.get("VERSION", "")
    if version not in ("3.12", "3.13", "3.14"):
        raise ValueError(f"Supported target Python versions are 3.12, 3.13 and 3.14; got {version!r}")
    if variables.get("Py_GIL_DISABLED") or variables.get("Py_DEBUG"):
        raise ValueError("Debug and free-threaded target Python builds are not supported")
    digits = version.replace(".", "")
    if variables.get("SOABI") != f"cpython-{digits}-riscv64-linux-gnu":
        raise ValueError("Target Python must use the CPython riscv64-linux-gnu ABI")
    include_dir = root / "include" / f"python{version}"
    for header in ("Python.h", "pyconfig.h", "patchlevel.h"):
        if not (include_dir / header).is_file():
            raise ValueError(f"Missing target Python header: {include_dir / header}")
    patchlevel = (include_dir / "patchlevel.h").read_text(encoding="utf-8")
    if not re.search(rf'#define\s+PY_VERSION\s+"{re.escape(version)}\.', patchlevel):
        raise ValueError(f"Target Python headers do not match Python {version}")
    return {
        "root": str(root),
        "include_dir": str(include_dir),
        "version": version,
        "python_tag": f"cp{digits}",
        "abi_tag": f"cp{digits}",
    }


def load_target_numpy(target_python):
    """Find NumPy headers and validate wheel metadata without importing NumPy."""
    packages = Path(target_python["root"]) / "lib" / f"python{target_python['version']}" / "site-packages"
    metadata_dirs = list(packages.glob("numpy-*.dist-info"))
    include_dir = packages / "numpy" / "_core" / "include"
    if len(metadata_dirs) != 1 or not (include_dir / "numpy" / "arrayobject.h").is_file():
        raise ValueError(f"Install one target NumPy >= 2.2 wheel into {packages}")
    metadata_dir = metadata_dirs[0]
    metadata = Parser().parsestr((metadata_dir / "METADATA").read_text(encoding="utf-8"))
    version = metadata.get("Version", "")
    match = re.match(r"(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (2, 2):
        raise ValueError(f"Target NumPy >= 2.2 is required; found {version!r}")
    wheel = Parser().parsestr((metadata_dir / "WHEEL").read_text(encoding="utf-8"))
    prefix = f"{target_python['python_tag']}-{target_python['abi_tag']}-"
    if not any(
        tag.startswith(prefix) and all(platform.endswith("_riscv64") for platform in tag[len(prefix) :].split("."))
        for tag in wheel.get_all("Tag", [])
    ):
        raise ValueError(f"Target NumPy must have a {prefix}*riscv64 wheel tag")
    return {"include_dir": str(include_dir), "version": version}
