#!/usr/bin/env python3
"""Build and check the serial MFEM #5535 prototype without MPI or servers."""

import argparse
import datetime
import json
import pathlib
import resource
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--build", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent
    source, build = args.source.resolve(), args.build.resolve()
    output = (args.output or root / "results").resolve()
    output.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    records = []

    def run(label, command, expect_failure=False):
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (output / (label + ".log")).write_text(result.stdout)
        records.append({"label": label, "command": command,
                        "returncode": result.returncode})
        if expect_failure:
            if (result.returncode == 0 or
                    "Gradient block mask must be empty" not in result.stdout):
                raise RuntimeError(f"{label}: expected mask validation failure")
        elif result.returncode:
            raise RuntimeError(f"{label} failed; see {output / (label + '.log')}")
        print(f"{label}: {'expected rejection' if expect_failure else 'PASS'}")
        return result.stdout

    try:
        run("configure", [
            "cmake", "-S", str(source), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Debug", "-DCMAKE_CXX_FLAGS_DEBUG=-O0 -g0",
            "-DMFEM_USE_MPI=OFF", "-DMFEM_USE_METIS=OFF",
            "-DMFEM_USE_LAPACK=OFF", "-DMFEM_USE_OPENMP=OFF",
            "-DMFEM_USE_ZLIB=OFF", "-DMFEM_USE_GNUTLS=OFF",
            "-DMFEM_ENABLE_EXAMPLES=OFF", "-DMFEM_ENABLE_MINIAPPS=OFF",
            "-DMFEM_ENABLE_TESTING=OFF"])
        run("build", ["cmake", "--build", str(build), "--target", "mfem",
                      "-j", str(args.jobs)])
        executable = build / "selective_test"
        run("compile", ["c++", "-std=c++17", "-O0", "-g0",
                        "-I" + str(build), "-I" + str(source),
                        str(root / "selective_test.cpp"),
                        str(build / "libmfem.a"), "-o", str(executable)])
        print(run("regression", [str(executable)]), end="")
        for rows, cols in ((1, 1), (0, 2), (2, 0), (3, 3)):
            run(f"invalid-mask-{rows}x{cols}",
                [str(executable), "--invalid-mask", str(rows), str(cols)],
                expect_failure=True)
    finally:
        (output / "commands.json").write_text(json.dumps({
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source": str(source), "build": str(build), "commands": records,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
