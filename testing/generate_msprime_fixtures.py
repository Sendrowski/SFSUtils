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

# VCF-Zarr fixture for the zarr-backend tests, built from the same VCF via the bio2zarr Python API.
# The API takes a plain (unindexed) VCF, so no bgzip/tabix is needed.
import shutil
from bio2zarr import vcf as _bio2zarr_vcf
shutil.rmtree(f"{OUT}/two_epoch.vcz", ignore_errors=True)
_bio2zarr_vcf.convert([f"{OUT}/two_epoch.vcf"], f"{OUT}/two_epoch.vcz")
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


# --- joint (three-population) fixture -------------------------------------------------------------
# A, B and C from two successive splits; tskit's joint AFS over three sample sets is the exact
# ground-truth three-dimensional joint SFS. Small per-population sample sizes keep the array compact.
THREE_POP_SEED = 7

demography_three = msprime.Demography()
demography_three.add_population(name="A", initial_size=10_000)
demography_three.add_population(name="B", initial_size=5_000)
demography_three.add_population(name="C", initial_size=8_000)
demography_three.add_population(name="BC", initial_size=10_000)
demography_three.add_population(name="ANC", initial_size=10_000)
demography_three.add_population_split(time=2_000, derived=["B", "C"], ancestral="BC")
demography_three.add_population_split(time=4_000, derived=["A", "BC"], ancestral="ANC")

ts_three = msprime.sim_ancestry(
    samples={"A": 3, "B": 3, "C": 2},  # -> 6, 6, 4 haplotypes
    demography=demography_three,
    sequence_length=2e6,
    recombination_rate=1e-8,
    random_seed=THREE_POP_SEED,
)
ts_three = msprime.sim_mutations(ts_three, rate=1e-8, random_seed=THREE_POP_SEED)

non_biallelic_three = [
    s.id for s in ts_three.sites()
    if len({s.ancestral_state} | {m.derived_state for m in s.mutations}) != 2
]
ts_three = ts_three.delete_sites(non_biallelic_three)

pop_ids_three = {p.metadata["name"]: p.id for p in ts_three.populations() if p.metadata.get("name")}
nodes_three = {name: ts_three.samples(population=pop_ids_three[name]) for name in ("A", "B", "C")}

jsfs_three = ts_three.allele_frequency_spectrum(
    sample_sets=[list(nodes_three[name]) for name in ("A", "B", "C")],
    polarised=True, span_normalise=False,
).astype(int)

ind_pop_three = {}
for ind in ts_three.individuals():
    ind_pop_three.setdefault(ts_three.node(ind.nodes[0]).population, []).append(f"tsk_{ind.id}")
pops_three = {name: ind_pop_three[pop_ids_three[name]] for name in ("A", "B", "C")}
n_per_pop_three = {name: len(nodes_three[name]) for name in ("A", "B", "C")}

with open(f"{OUT}/three_pop_joint.vcf", "w") as f:
    ts_three.write_vcf(f)

JointSFS(jsfs_three, pop_names=["A", "B", "C"]).to_file(f"{OUT}/three_pop_joint.jsfs.json")

with open(f"{OUT}/three_pop_joint.pops.json", "w") as f:
    json.dump({"pops": pops_three, "n": n_per_pop_three}, f, indent=2)

print()
print(f"three-pop haplotypes: {n_per_pop_three}")
print(f"three-pop segregating sites: {ts_three.num_sites}")
print(f"three-pop SFS shape: {jsfs_three.shape}, sum: {jsfs_three.sum()}")
print(f"wrote three-pop joint fixtures to {OUT}/")


# --- two-SFS (two-site) fixture -------------------------------------------------------------------
# A single recombining sequence gives linked pairs of segregating sites; the two-SFS counts pairs
# within a genomic-distance window. The reference matrix is built here by an independent naive
# double loop that reads the written VCF back with cyvcf2 (a different code path from the parser's
# streaming sliding-window accumulation), so the test genuinely validates the parser.
import cyvcf2

from sfsutils.spectrum import TwoSFS

TWO_SFS_SEED = 42
TWO_SFS_DISTANCE = 2_000
TWO_SFS_OFFSET = 0

ts_two = msprime.sim_ancestry(
    samples=10,  # -> 20 haplotypes
    population_size=10_000,
    sequence_length=2e6,
    recombination_rate=1e-8,
    random_seed=TWO_SFS_SEED,
)
ts_two = msprime.sim_mutations(ts_two, rate=1.5e-8, random_seed=TWO_SFS_SEED)

non_biallelic_two = [
    s.id for s in ts_two.sites()
    if len({s.ancestral_state} | {m.derived_state for m in s.mutations}) != 2
]
ts_two = ts_two.delete_sites(non_biallelic_two)

n_two = ts_two.num_samples  # 20 haplotypes

with open(f"{OUT}/two_sfs.vcf", "w") as f:
    ts_two.write_vcf(f)

# read positions and derived-allele counts back from the VCF (allele 0 = REF = ancestral)
sites = [
    (v.POS, sum(a for gt in v.genotypes for a in gt[:2] if a > 0))
    for v in cyvcf2.VCF(f"{OUT}/two_sfs.vcf")
]

# reference two-SFS: naive forward-pair count within the window, then symmetrized
max_distance = TWO_SFS_OFFSET + TWO_SFS_DISTANCE
ref = np.zeros((n_two + 1, n_two + 1))
for a in range(len(sites)):
    pos_a, der_a = sites[a]
    for b in range(a + 1, len(sites)):
        distance = sites[b][0] - pos_a
        if distance > max_distance:
            break  # positions are sorted, so no later site is closer
        if TWO_SFS_OFFSET < distance:
            ref[der_a, sites[b][1]] += 1
ref = (ref + ref.T) / 2

TwoSFS(ref).to_file(f"{OUT}/two_sfs.ref.json")

with open(f"{OUT}/two_sfs.meta.json", "w") as f:
    json.dump({"n": n_two, "distance": TWO_SFS_DISTANCE, "offset": TWO_SFS_OFFSET}, f, indent=2)

print()
print(f"two-SFS haplotypes: {n_two}")
print(f"two-SFS segregating sites: {ts_two.num_sites}")
print(f"two-SFS total pairs (window {TWO_SFS_DISTANCE} bp): {int(ref.sum())}")
print(f"wrote two-SFS fixtures to {OUT}/")
