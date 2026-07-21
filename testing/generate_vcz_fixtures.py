"""
Generate the VCF-Zarr interoperability fixture ``resources/msprime/typed_info.vcz`` (used by
``test_vcz_interop.py`` to check we read a bio2zarr store's typed INFO fields). Built from a small
inline VCF via the bio2zarr Python API, which takes a plain (unindexed) VCF, so no bgzip/tabix is
needed. Run from the repository root: ``python testing/generate_vcz_fixtures.py``.
"""
import shutil
import tempfile
from pathlib import Path

from bio2zarr import vcf as bio2zarr_vcf

OUT = "resources/msprime/typed_info.vcz"

# a String, a Float and an Integer INFO field, so the reader can be checked for each type
VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=100>
##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">
##INFO=<ID=AA_prob,Number=1,Type=Float,Description="ancestral prob">
##INFO=<ID=DP,Number=1,Type=Integer,Description="depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2
1\t10\t.\tA\tT\t.\t.\tAA=A;AA_prob=0.9;DP=30\tGT\t0|1\t1|1
1\t20\t.\tC\tG\t.\t.\tAA=C;AA_prob=0.8;DP=25\tGT\t0|0\t0|1
"""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        vcf = Path(tmp) / "typed_info.vcf"
        vcf.write_text(VCF)
        shutil.rmtree(OUT, ignore_errors=True)
        bio2zarr_vcf.convert([str(vcf)], OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
