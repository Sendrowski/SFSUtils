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

import gzip
import os
os.makedirs(OUT, exist_ok=True)
ts.dump(f"{OUT}/two_epoch.trees")
with open(f"{OUT}/two_epoch.vcf", "w") as f:
    ts.write_vcf(f)
# a placeholder reference over the single contig ("1"), so the TargetSiteCounter has a FASTA to
# satisfy its genome-length requirement; the bases are irrelevant to target-site accounting
with gzip.open(f"{OUT}/two_epoch.ref.fasta.gz", "wt") as fa:
    fa.write(">1\n" + "A" * int(ts.sequence_length) + "\n")
np.savetxt(f"{OUT}/two_epoch.sfs.txt", sfs, fmt="%d",
           header=f"unfolded SFS (bins 0..{n}) from tskit; two-epoch demography, seed {SEED}")

print(f"n haplotypes: {n}")
print(f"segregating sites: {ts.num_sites}")
print(f"polymorphic SFS (1..{n-1}): {sfs[1:n].tolist()}")
print(f"wrote fixtures to {OUT}/")


# --- joint (two-population) fixture ---------------------------------------------------------------
# A and B split from a common ancestor with symmetric migration; tskit's joint AFS (with one sample
# set per population) is the exact ground-truth joint SFS, and the VCF again encodes the ancestral
# allele as REF so the parser recovers it with skip_non_polarized=False.
import json

from sfsutils.spectrum import JointSFS

JOINT_SEED = 42

demography_joint = msprime.Demography()
demography_joint.add_population(name="A", initial_size=10_000)
demography_joint.add_population(name="B", initial_size=5_000)
demography_joint.add_population(name="ANC", initial_size=10_000)
demography_joint.add_population_split(time=3_000, derived=["A", "B"], ancestral="ANC")
demography_joint.set_migration_rate("A", "B", 1e-4)
demography_joint.set_migration_rate("B", "A", 1e-4)

ts_joint = msprime.sim_ancestry(
    samples={"A": 4, "B": 3},  # -> 8 and 6 haplotypes
    demography=demography_joint,
    sequence_length=2e6,
    recombination_rate=1e-8,
    random_seed=JOINT_SEED,
)
ts_joint = msprime.sim_mutations(ts_joint, rate=1e-8, random_seed=JOINT_SEED)

# keep only strictly biallelic sites (as for the single-population fixture)
non_biallelic_joint = [
    s.id for s in ts_joint.sites()
    if len({s.ancestral_state} | {m.derived_state for m in s.mutations}) != 2
]
ts_joint = ts_joint.delete_sites(non_biallelic_joint)

# population id per name and the corresponding sample (haplotype) nodes
pop_ids = {p.metadata["name"]: p.id for p in ts_joint.populations() if p.metadata.get("name")}
nodes = {name: ts_joint.samples(population=pid) for name, pid in pop_ids.items() if name in ("A", "B")}

# exact ground-truth joint SFS from tskit (axis order A, B)
jsfs = ts_joint.allele_frequency_spectrum(
    sample_sets=[list(nodes["A"]), list(nodes["B"])], polarised=True, span_normalise=False
).astype(int)

# map diploid individuals (VCF sample names tsk_i) to their population
ind_pop = {}
for ind in ts_joint.individuals():
    ind_pop.setdefault(ts_joint.node(ind.nodes[0]).population, []).append(f"tsk_{ind.id}")
pops = {name: ind_pop[pid] for name, pid in pop_ids.items() if name in ("A", "B")}
n_per_pop = {name: len(nodes[name]) for name in ("A", "B")}

with open(f"{OUT}/two_epoch_joint.vcf", "w") as f:
    ts_joint.write_vcf(f)

JointSFS(jsfs, pop_names=["A", "B"]).to_file(f"{OUT}/two_epoch_joint.jsfs.json")

with open(f"{OUT}/two_epoch_joint.pops.json", "w") as f:
    json.dump({"pops": pops, "n": n_per_pop}, f, indent=2)

print()
print(f"joint haplotypes: A={n_per_pop['A']}, B={n_per_pop['B']}")
print(f"joint segregating sites: {ts_joint.num_sites}")
print(f"joint SFS shape: {jsfs.shape}, sum: {jsfs.sum()}")
print(f"wrote joint fixtures to {OUT}/")
