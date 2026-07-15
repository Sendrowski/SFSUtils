import sfsutils as sf

# parse selected and neutral SFS from human chromosome 1
p = sf.Parser(
    vcf="https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/"
        "hgdp_wgs.20190516.full.chr21.vcf.gz",
    fasta="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
          "dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz",
    gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
        "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
    aliases=dict(chr21=['21']),
    n=10,
    target_site_counter=sf.TargetSiteCounter(
        n_samples=1000000,
        n_target_sites=sf.Annotation.count_target_sites(
            "http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
            "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz"
        )['21']
    ),
    annotations=[
        sf.DegeneracyAnnotation()
    ],
    filtrations=[
        sf.CodingSequenceFiltration()
    ],
    stratifications=[sf.DegeneracyStratification()],
    info_ancestral='AA_ensembl'
)

sfs = p.parse()

sfs.plot()

pass
