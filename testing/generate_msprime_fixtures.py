"""
Generate the fixtures for the msprime-based VCF-parsing validation test.

A tree sequence is simulated under a two-epoch demography, its exact unfolded SFS is taken from
tskit, and the genotypes are written to a VCF. Because tskit writes the ancestral allele (allele 0)
as REF, the parser can recover the correct polarisation via ``skip_non_polarized=False`` (REF as
ancestral) without an ``AA`` tag. The test itself only needs sfsutils + numpy, so it runs in CI
where msprime/tskit are absent; this script is run manually to (re)generate the committed fixtures.

Usage (in the sfsutils-dev env, from the repository root):
    python testing/generate_msprime_fixtures.py
"""
import numpy as np
import msprime

OUT = "resources/msprime"
SEED = 42
N_DIPLOID = 10  # -> 20 haplotypes

# --- two-epoch demography: ancestral expansion (bottleneck looking forward in time) ---
demography = msprime.Demography()
demography.add_population(name="A", initial_size=10_000)
demography.add_population_parameters_change(time=2_000, initial_size=2_000, population="A")

ts = msprime.sim_ancestry(
    samples=N_DIPLOID,
    demography=demography,
    sequence_length=1e6,
    recombination_rate=1e-8,
    random_seed=SEED,
)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=SEED + 1)

# keep only strictly biallelic sites so tskit's AFS and the VCF-derived SFS agree exactly
# (the discrete-genome model can otherwise place recurrent/multiallelic mutations at a site)
non_biallelic = [
    s.id for s in ts.sites()
    if len({s.ancestral_state} | {m.derived_state for m in s.mutations}) != 2
]
ts = ts.delete_sites(non_biallelic)

n = ts.num_samples  # 20 haplotypes
# exact unfolded SFS (counts per derived-allele count, bins 0..n)
sfs = ts.allele_frequency_spectrum(polarised=True, span_normalise=False).astype(int)

import os
os.makedirs(OUT, exist_ok=True)
ts.dump(f"{OUT}/two_epoch.trees")
with open(f"{OUT}/two_epoch.vcf", "w") as f:
    ts.write_vcf(f)
np.savetxt(f"{OUT}/two_epoch.sfs.txt", sfs, fmt="%d",
           header=f"unfolded SFS (bins 0..{n}) from tskit; two-epoch demography, seed {SEED}")

print(f"n haplotypes: {n}")
print(f"segregating sites: {ts.num_sites}")
print(f"polymorphic SFS (1..{n-1}): {sfs[1:n].tolist()}")
print(f"wrote fixtures to {OUT}/")
