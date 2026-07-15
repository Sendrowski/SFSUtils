import sfsutils as sf

basepath = "../resources/genome/betula/"

# instantiate parser
p = sf.Parser(
    n=8,  # SFS sample size
    vcf=(basepath + "all.with_outgroups.vcf.gz"),
    fasta=basepath + "genome.subset.1000.fasta.gz",
    gff=basepath + "genome.gff.gz",
    annotations=[
        sf.DegeneracyAnnotation(),  # determine degeneracy
        sf.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730"],  # use one outgroup
            n_ingroups=20,  # subsample size
            max_sites=50000
        )
    ],
    stratifications=[sf.DegeneracyStratification()]
)

# obtain SFS
spectra: sf.Spectra = p.parse()

spectra.plot(title="SFS")

pass
