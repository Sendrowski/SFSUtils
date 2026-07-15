# example for ancestral allele annotation
import sfsutils as sf

ann = sf.Annotator(
    vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
    annotations=[sf.MaximumLikelihoodAncestralAnnotation(
        outgroups=["ERR2103730"],
        n_ingroups=15
    )],
    output="genome.polarized.vcf.gz"
)

ann.annotate()

pass
