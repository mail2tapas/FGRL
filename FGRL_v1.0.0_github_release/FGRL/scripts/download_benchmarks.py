"""Download the Pegasus benchmark workflows (Montage, CyberShake, Epigenomics) used in the paper.

    python scripts/download_benchmarks.py            # into data/pegasus/

The DAX files are the synthetic-but-trace-calibrated benchmark workflows produced by the
Pegasus Workflow Generator (Bharathi et al., 2008; Juve et al., 2013) and distributed with
WorkflowSim (Chen & Deelman, 2012). Each file is verified against the SHA-256 recorded
below, so every run uses byte-identical inputs.
"""
import hashlib
import os
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/WorkflowSim/WorkflowSim-1.0/master/config/dax/"
FILES = {
    "CyberShake_100.xml": "183ac79c4ee80a6293adff71d8386e7538d31a9bd1b609b176d94b131b9fdb52",
    "CyberShake_1000.xml": "4314ae0e6bb43c3f74818306b600151c0614b62184295438350837217f1df95b",
    "CyberShake_30.xml": "230b9861c69cb9b04165b6d37952a39059aaa1a406350f6c03b238c81aa0228e",
    "CyberShake_50.xml": "9fbf8c6b6455abf74b4a5cd4800af76bdafcc0a13a210ab8ad3055166b228bc2",
    "Epigenomics_100.xml": "374521746417b18133682de21b654b84c32acde6332462c0ce916c4ba12c7f36",
    "Epigenomics_24.xml": "867696c4d3028eded842bec41c9ce3ef6ddceb3ccd925212f1282281101c7af1",
    "Epigenomics_46.xml": "994c4bea6c5709bcf7cc8f703a53eab9cbb9af049cf331c02735b8cd115040ed",
    "Epigenomics_997.xml": "e9fecca3453f31873e890c36afe7c6ef87f53f786bf70c39a011b68a1da7a2e4",
    "Montage_100.xml": "ac26318b5d34e8d4afa743ab6735d4daa2e8163b3493e5d640394f771220c6bf",
    "Montage_1000.xml": "720dfe953dcf54958f33ba33d2bdac8563c306081ff8a601ab6d14bdaeea70f5",
    "Montage_25.xml": "3c791f17b8e57a06e5e7afad57fffc625c0611caf168036efeaa18b1888bab19",
    "Montage_50.xml": "7fcbdcdc89c6d5c6afe0a86a0900b13a1c6debd2139033949bb7296539963454",
}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pegasus")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = True
    for name, h in FILES.items():
        path = os.path.join(OUT, name)
        if os.path.exists(path) and sha(open(path, "rb").read()) == h:
            print(f"ok      {name}")
            continue
        data = urllib.request.urlopen(BASE + name, timeout=60).read()
        if sha(data) != h:
            print(f"MISMATCH {name}: expected {h}, got {sha(data)}"); ok = False; continue
        open(path, "wb").write(data)
        print(f"fetched {name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
