"""
Generate the synthetic all-sites datasets used by the ``2-Site SFS`` documentation notebook.

For each coalescent model (the Kingman coalescent and a Beta multiple-merger coalescent) many independent
tightly-linked contigs are simulated and written as separate contigs of one all-sites VCF (every position emitted,
monomorphic sites as the zero-derived class). Pooling many contigs gives the two-SFS branch-length correlation
(:meth:`~sfsutils.spectrum.TwoSFS.corr`) a clean estimate over many independent genealogical realizations. The
contigs are tightly linked (no within-contig recombination), which is the regime where the branch-length
correlation is strongest; recombination decorrelates the two sites toward zero. The monomorphic sites are required:
they anchor the marginal to the site-frequency spectrum, without which the branch-length covariance/correlation is
undefined.

The simulations use coalescent units (population_size=1): msprime's Beta coalescent matches PhaseGen's Lambda-Beta
coalescent only at this scale, and a larger effective size changes its multiple-merger genealogy (flipping the
low-frequency sign). The two models are emitted at different mutation rates to reach a comparable number of
segregating sites. The low-frequency off-diagonal correlations then contrast by model: under the Kingman coalescent
adjacent low-frequency classes are negatively correlated (Fu 1995), whereas under the Beta coalescent the multiple
mergers make them positively correlated (Birkner et al. 2013).

Usage (in the sfsutils-dev env, from the repository root):
    python testing/generate_two_sfs_demo.py
"""
import gzip
import io

import msprime
import numpy as np

OUT = "resources/msprime"
N_DIPLOID = 5          # -> 10 haplotypes
NE = 1                 # coalescent units (population_size=1): msprime's Beta coalescent matches PhaseGen only at
                       # this scale; larger Ne changes its multiple-merger genealogy and flips the low-frequency sign
N_CONTIGS = 600        # independent tightly-linked contigs pooled into one all-sites VCF (more = cleaner estimate)
L = 1000               # length of each contig (all positions emitted)
R = 0                  # tightly linked: each contig is one genealogy; pooling many contigs gives a clean estimate
ALPHA = 1.3            # Beta(alpha) multiple-merger strength; smaller alpha = stronger mergers, more positive signal
MU_KINGMAN = 5e-3      # per-base mutation rate under the Kingman coalescent (~3% polymorphic at Ne=1)
MU_BETA = 1.4e-3       # lower under Beta(1.3): its longer total branch length at Ne=1 reaches ~3% at a lower rate


def _simulate_contig(model, mu, seed):
    """Simulate one tightly-linked contig, place mutations and keep strictly biallelic sites."""
    # haploid (ploidy=1): msprime's Beta coalescent under diploidy is a different multiple-merger model, whereas
    # the haploid one matches PhaseGen's Lambda-Beta coalescent; 2*N_DIPLOID haplotypes are paired into VCF diploids
    ts = msprime.sim_ancestry(samples=2 * N_DIPLOID, ploidy=1, population_size=NE, sequence_length=L,
                              recombination_rate=R, model=model, random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=mu, discrete_genome=True, random_seed=seed + 1)

    # a discrete genome can place recurrent mutations at one position; keep only strictly biallelic sites
    non_biallelic = [s.id for s in ts.sites()
                     if len({s.ancestral_state} | {m.derived_state for m in s.mutations}) != 2]

    return ts.delete_sites(non_biallelic)


def _write_all_sites_vcf(contigs, path):
    """
    Write every position of every contig as a VCF record. Segregating sites carry their genotypes with REF=A
    (ancestral) and ALT=C (derived); monomorphic sites are REF=A, ALT=``.`` with all-``0/0`` genotypes. Each
    contig is its own ``##contig`` so the parser resets the two-SFS pairing at contig boundaries.
    """
    n_samples = N_DIPLOID  # diploid VCF samples
    sample_names = [f"tsk_{i}" for i in range(n_samples)]

    contig_lines = "".join(f"##contig=<ID=c{i}>\n" for i in range(len(contigs)))
    header = (
        "##fileformat=VCFv4.2\n" + contig_lines +
        '##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sample_names)
    )

    mono_tail = "\t".join(["0/0"] * n_samples)
    n_snps = 0
    buf = io.StringIO()
    buf.write(header + "\n")

    for ci, ts in enumerate(contigs):
        chrom = f"c{ci}"
        genotypes = ts.genotype_matrix()  # (num_sites, n), entries 0/1
        positions = ts.tables.sites.position.astype(int)

        # 0-based position -> site index, dropping sites that collapse onto the same integer position
        site_at = {}
        for j, p in enumerate(positions):
            site_at.setdefault(int(p), j)
        n_snps += len(site_at)

        for pos0 in range(L):
            j = site_at.get(pos0)
            if j is None:
                buf.write(f"{chrom}\t{pos0 + 1}\t.\tA\t.\t.\t.\tAA=A\tGT\t{mono_tail}\n")
            else:
                g = genotypes[j]
                gts = "\t".join(f"{g[2 * k]}/{g[2 * k + 1]}" for k in range(n_samples))
                buf.write(f"{chrom}\t{pos0 + 1}\t.\tA\tC\t.\t.\tAA=A\tGT\t{gts}\n")

    with gzip.open(path, "wt") as f:
        f.write(buf.getvalue())

    return n_snps


def generate(model, seed, path):
    """Simulate ``N_CONTIGS`` independent contigs and write them as one all-sites VCF."""
    rng = np.random.default_rng(seed)
    mu = MU_BETA if isinstance(model, msprime.BetaCoalescent) else MU_KINGMAN

    contigs = [_simulate_contig(model, mu, int(rng.integers(1, 2 ** 31))) for _ in range(N_CONTIGS)]

    return _write_all_sites_vcf(contigs, path)


if __name__ == "__main__":
    import logging

    import sfsutils as su
    from sfsutils.settings import Settings

    Settings.disable_pbar = True
    logging.getLogger("sfsutils").setLevel(logging.WARNING)

    jobs = [
        ("kingman", msprime.StandardCoalescent(), 20, f"{OUT}/two_sfs_kingman.all.vcf.gz"),
        (f"beta-{ALPHA}", msprime.BetaCoalescent(alpha=ALPHA), 77, f"{OUT}/two_sfs_beta.all.vcf.gz"),
    ]

    for name, model, seed, path in jobs:
        n_snps = generate(model, seed, path)
        print(f"{name}: wrote {n_snps} segregating sites to {path}", flush=True)

        sfs2 = su.Parser(vcf=path, n=N_DIPLOID * 2, two_sfs=True, d=1000).parse()
        block = sfs2.corr().data[1:5, 1:5]
        mono_frac = 1 - n_snps / (N_CONTIGS * L)
        print(f"{name}: {n_snps} SNPs, monomorphic fraction {mono_frac:.4f}", flush=True)
        print(np.array2string(np.round(block, 3)), flush=True)
