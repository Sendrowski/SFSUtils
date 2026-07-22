"""
Command-line interface for ``sfsutils``.

Three subcommands mirror the package's main operations: ``parse`` derives a (one-dimensional, joint, or
two-site) site-frequency spectrum from a VCF, a VCF-Zarr store, or a tskit tree sequence; ``filter`` writes a
filtered VCF; and ``annotate`` writes an annotated VCF. Each subcommand instantiates the corresponding class
and calls its method; short option names map to the underlying stratification, annotation, and filtration
classes.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"

import argparse
import logging
import os
import sys
from typing import Callable, Dict, List, Sequence

from . import __version__

logger = logging.getLogger('sfsutils')


def _split_csv(value: str) -> List[str]:
    """
    Split a comma-separated option value into a list of non-empty tokens.

    :param value: The raw option string.
    :return: List of tokens.
    """
    return [token.strip() for token in value.split(',') if token.strip()]


def _parse_pops(value: str) -> Dict[str, List[str]]:
    """
    Parse a population specification of the form ``A=s1,s2;B=s3,s4`` into a mapping.

    :param value: The raw option string.
    :return: Mapping of population name to sample names.
    :raises argparse.ArgumentTypeError: If a group is malformed, names a population twice, or leaves a name or
        its sample list empty.
    """
    pops: Dict[str, List[str]] = {}

    for group in value.split(';'):
        if not group.strip():
            continue

        name, sep, samples = group.partition('=')

        if not sep:
            raise argparse.ArgumentTypeError(f"Invalid population spec '{group}'; expected 'name=sample1,sample2'.")

        name = name.strip()

        if not name:
            raise argparse.ArgumentTypeError(f"Invalid population spec '{group}'; the population name is empty.")

        # a repeated name would overwrite the earlier group and yield a joint SFS of the wrong dimension, which
        # looks entirely valid downstream
        if name in pops:
            raise argparse.ArgumentTypeError(f"Population '{name}' is specified more than once.")

        pops[name] = _split_csv(samples)

        if not pops[name]:
            raise argparse.ArgumentTypeError(f"Population '{name}' has no samples.")

    return pops


def _positive_int(value: str) -> int:
    """
    Parse an option value that has to be a positive integer.

    :param value: The raw option string.
    :return: The parsed integer.
    :raises argparse.ArgumentTypeError: If the value is not an integer of at least one.
    """
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value '{value}'.")

    # the library stop conditions compare for equality, so a limit below one never fires and is silently
    # equivalent to no limit at all
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Value must be at least 1, got {parsed}.")

    return parsed


def _sample_size(value: str) -> int:
    """
    Parse an option value that has to be an SFS sample size.

    :param value: The raw option string.
    :return: The parsed integer.
    :raises argparse.ArgumentTypeError: If the value is not an integer of at least two.
    """
    parsed = _positive_int(value)

    # bin 1 is the divergence bin for n == 1, so a spectrum of that size books every segregating site as a
    # fixed difference
    if parsed < 2:
        raise argparse.ArgumentTypeError(f"The sample size must be at least 2, got {parsed}.")

    return parsed


def _configure_logging(verbose: int, quiet: bool) -> None:
    """
    Set the package logger level from the verbosity flags.

    :param verbose: Verbosity count (``-v`` -> DEBUG).
    :param quiet: Whether to silence INFO logs.
    """
    if quiet:
        logger.setLevel(logging.WARNING)
    elif verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


# --- short-name -> factory maps for the parser-configuration objects ------------------------------

def _build_stratifications(names: List[str], contigs: List[str] | None = None) -> list:
    """
    Build stratification instances from short names.

    :param names: Short stratification names.
    :param contigs: Contigs to stratify by (only for the ``contig`` stratification).
    :return: List of stratification instances.
    :raises SystemExit: On an unknown name.
    """
    from . import (DegeneracyStratification, SynonymyStratification, BaseTransitionStratification,
                   TransitionTransversionStratification, AncestralBaseStratification, ContigStratification)

    factories: Dict[str, Callable[[], object]] = {
        'degeneracy': DegeneracyStratification,
        'synonymy': SynonymyStratification,
        'base-transition': BaseTransitionStratification,
        'transition-transversion': TransitionTransversionStratification,
        'ancestral-base': AncestralBaseStratification,
        'contig': lambda: ContigStratification(contigs=contigs),
    }

    return [_lookup(factories, name, 'stratification')() for name in names]


def _build_filtrations(names: List[str], contigs: List[str] | None) -> list:
    """
    Build filtration instances from short names.

    :param names: Short filtration names.
    :param contigs: Contigs to keep (only for the ``contig`` filtration).
    :return: List of filtration instances.
    :raises SystemExit: On an unknown name.
    """
    from . import (SNPFiltration, SNVFiltration, PolyAllelicFiltration, CodingSequenceFiltration, CpGFiltration,
                   ContigFiltration, NoFiltration, AllFiltration)

    def _contig():
        if not contigs:
            raise SystemExit("The 'contig' filtration requires --contigs.")

        return ContigFiltration(contigs=contigs)

    factories: Dict[str, Callable[[], object]] = {
        'snp': SNPFiltration,
        'snv': SNVFiltration,
        'poly-allelic': PolyAllelicFiltration,
        'coding-sequence': CodingSequenceFiltration,
        'cpg': CpGFiltration,
        'contig': _contig,
        'no': NoFiltration,
        'all': AllFiltration,
    }

    return [_lookup(factories, name, 'filtration')() for name in names]


def _build_annotations(names: List[str], outgroups: List[str] | None, n_ingroups: int) -> list:
    """
    Build annotation instances from short names.

    :param names: Short annotation names.
    :param outgroups: Outgroup samples (required for the maximum-likelihood ancestral annotation).
    :param n_ingroups: Minimum number of ingroups for the maximum-likelihood ancestral annotation.
    :return: List of annotation instances.
    :raises SystemExit: On an unknown name or missing outgroups.
    """
    from . import DegeneracyAnnotation, SynonymyAnnotation, MaximumLikelihoodAncestralAnnotation

    def _mle():
        if not outgroups:
            raise SystemExit("The 'maximum-likelihood-ancestral' annotation requires --outgroups.")

        return MaximumLikelihoodAncestralAnnotation(outgroups=outgroups, n_ingroups=n_ingroups)

    factories: Dict[str, Callable[[], object]] = {
        'degeneracy': DegeneracyAnnotation,
        'synonymy': SynonymyAnnotation,
        'maximum-likelihood-ancestral': _mle,
    }

    return [_lookup(factories, name, 'annotation')() for name in names]


def _lookup(factories: Dict[str, Callable], name: str, kind: str) -> Callable:
    """
    Look up a factory by name, raising a clear error listing the valid choices.

    :param factories: Mapping of name to factory.
    :param name: The requested name.
    :param kind: The kind of object (for the error message).
    :return: The factory.
    :raises SystemExit: If the name is unknown.
    """
    if name not in factories:
        raise SystemExit(f"Unknown {kind} '{name}'. Choose from: {', '.join(sorted(factories))}.")

    return factories[name]


# --- subcommand handlers --------------------------------------------------------------------------

def _run_parse(args: argparse.Namespace) -> int:
    """
    Derive an SFS from a VCF and write it to file.

    :param args: Parsed arguments.
    :return: Exit code.
    """
    from . import Parser

    spectra = Parser(
        source=_input_source(args),
        n=args.n,
        pops=args.pops,
        gff=args.gff,
        fasta=args.fasta,
        info_ancestral=args.info_ancestral,
        skip_non_polarized=args.skip_non_polarized,
        stratifications=_build_stratifications(args.stratify, args.contigs),
        annotations=_build_annotations(args.annotate, args.outgroups, args.n_ingroups),
        filtrations=_build_filtrations(args.filter, args.contigs),
        max_sites=args.max_sites if args.max_sites is not None else float("inf"),
        seed=args.seed,
        subsample_mode=args.subsample_mode,
        two_sfs=args.two_sfs,
        d=args.two_sfs_distance,
        two_sfs_offset=args.two_sfs_offset,
    ).parse()

    if spectra.is_empty:
        logger.error("parse: no sites were included in the spectra, so nothing was written to %s. Check that the "
                     "sample size does not exceed the input, and that the ancestral allele information the "
                     "polarization needs is present (or pass --no-skip-non-polarized).", args.output)
        return 1

    spectra.to_file(args.output)
    logger.info("parse: wrote spectrum to %s", args.output)

    return 0


def _run_filter(args: argparse.Namespace) -> int:
    """
    Filter a VCF and write the result.

    :param args: Parsed arguments.
    :return: Exit code.
    """
    from . import Filterer

    Filterer(
        source=_input_source(args),
        output=args.output,
        filtrations=_build_filtrations(args.filter, args.contigs),
        gff=args.gff,
        fasta=args.fasta,
        max_sites=args.max_sites if args.max_sites is not None else float("inf"),
    ).filter()

    logger.info("filter: wrote filtered sites to %s", args.output)

    return 0


def _run_annotate(args: argparse.Namespace) -> int:
    """
    Annotate a VCF and write the result.

    :param args: Parsed arguments.
    :return: Exit code.
    """
    from . import Annotator

    Annotator(
        source=_input_source(args),
        output=args.output,
        annotations=_build_annotations(args.annotation, args.outgroups, args.n_ingroups),
        gff=args.gff,
        fasta=args.fasta,
        info_ancestral=args.info_ancestral,
        max_sites=args.max_sites if args.max_sites is not None else float("inf"),
        seed=args.seed,
    ).annotate()

    logger.info("annotate: wrote annotated sites to %s", args.output)

    return 0


# --- argument parser ------------------------------------------------------------------------------

def _add_common_io(p: argparse.ArgumentParser, out_help: str) -> None:
    """
    Add the input-source group and ``--output`` shared by the ``filter`` and ``annotate`` subcommands.

    :param p: The subparser.
    :param out_help: Help text for ``--output``.
    """
    _add_input_source(p)
    p.add_argument("--output", required=True, help=out_help)


def _add_input_source(p: argparse.ArgumentParser) -> None:
    """
    Add a mutually exclusive, required input-source group accepting a VCF, a VCF-Zarr store, or a tskit
    tree sequence. The three map to the same underlying source; they are distinct flags for clarity.

    :param p: The subparser.
    """
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--vcf", help="Input VCF file (may be gzipped or a URL).")
    source.add_argument("--zarr", help="Input VCF-Zarr store (a .vcz or .zarr directory).")
    source.add_argument("--trees", help="Input tskit tree sequence (a .trees file).")


def _input_source(args: argparse.Namespace) -> str:
    """
    Resolve the selected input source from a VCF, VCF-Zarr, or tree-sequence flag.

    :param args: Parsed arguments.
    :return: The input source path.
    """
    return args.vcf or args.zarr or args.trees


def _check_output_distinct_from_input(args: argparse.Namespace) -> None:
    """
    Reject an ``--output`` path that resolves to the input source.

    :param args: Parsed arguments.
    :raises SystemExit: If the output is the input source, lies inside it, or is a directory holding it.
    """
    source, output = _input_source(args), args.output

    # a remote source is never written to, and neither path can be resolved when either is absent
    if not source or not output or '://' in source or not os.path.exists(source):
        return

    message = "Writing the output over the input would destroy it; choose a different output path."

    # the leaf need not exist yet, so the comparison is made against the nearest existing ancestor: an output
    # written inside a zarr store destroys the store just as an output equal to it does
    probe = os.path.abspath(output)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return
        probe = parent

    # samefile compares (st_dev, st_ino), so it catches a hard link and a case-insensitive filesystem, both
    # of which a string comparison of realpath() lets through
    if os.path.samefile(source, probe):
        raise SystemExit(f"--output '{output}' resolves to the input source '{source}', or to a path inside "
                         f"it. {message}")

    # a zarr output directory is emptied whole, taking an input stored below it with it
    if os.path.isdir(os.path.abspath(output)):
        ancestor = os.path.abspath(source)

        while os.path.dirname(ancestor) != ancestor:
            ancestor = os.path.dirname(ancestor)

            if os.path.samefile(ancestor, output):
                raise SystemExit(f"--output '{output}' is a directory containing the input source "
                                 f"'{source}'. {message}")


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level argument parser.

    :return: The argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="sfsutils",
        description="Derive site-frequency spectra from VCF files, and filter or annotate VCFs.",
    )
    parser.add_argument("--version", action="version", version=f"sfsutils {__version__}")

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="count", default=0,
                           help="Increase log verbosity (-v -> DEBUG).")
    verbosity.add_argument("-q", "--quiet", action="store_true",
                           help="Silence INFO logs (only WARNING and above).")

    sub = parser.add_subparsers(dest="command", required=True, metavar="{parse,filter,annotate}")

    _add_parse_parser(sub)
    _add_filter_parser(sub)
    _add_annotate_parser(sub)

    return parser


def _add_parse_parser(sub: argparse._SubParsersAction) -> None:
    """
    Register the ``parse`` subcommand.

    :param sub: The subparsers action.
    """
    p = sub.add_parser("parse", help="Derive an SFS from a VCF, VCF-Zarr store, or tree sequence.",
                       description="Derive a one-dimensional, joint (multi-population), or two-site SFS from a VCF, "
                                   "VCF-Zarr store, or tskit tree sequence.")
    _add_input_source(p)
    p.add_argument("--output", required=True,
                   help="Output spectrum file (CSV for a single-population SFS, JSON for a joint or two-site SFS).")

    p.add_argument("--n", type=_sample_size, required=True,
                   help="SFS sample size (per population for a joint SFS).")
    p.add_argument("--pops", type=_parse_pops, default=None,
                   help="Population spec for a joint SFS, e.g. 'A=s1,s2;B=s3,s4'.")
    p.add_argument("--fasta", default=None, help="FASTA reference (required by some annotations/filtrations).")
    p.add_argument("--gff", default=None, help="GFF annotation (required by some annotations/filtrations).")
    p.add_argument("--stratify", type=_split_csv, default=[],
                   help="Comma-separated stratifications (e.g. degeneracy,synonymy). Default: none.")
    p.add_argument("--annotate", type=_split_csv, default=[],
                   help="Comma-separated on-the-fly annotations (e.g. degeneracy, synonymy). Default: none.")
    p.add_argument("--filter", type=_split_csv, default=["poly-allelic"],
                   help="Comma-separated filtrations. Default: poly-allelic.")
    p.add_argument("--info-ancestral", default="AA", help="INFO tag holding the ancestral allele. Default: AA.")
    p.add_argument("--skip-non-polarized", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip sites without a valid ancestral tag. Pass --no-skip-non-polarized to use the "
                        "reference allele as ancestral for those sites instead. Default: enabled.")
    p.add_argument("--subsample-mode", choices=["random", "probabilistic"], default="probabilistic",
                   help="Down-sampling mode. Default: probabilistic.")
    p.add_argument("--two-sfs", action="store_true", help="Parse the two-site (2-D) SFS instead.")
    p.add_argument("--two-sfs-distance", type=_positive_int, default=1000,
                   help="Width in bp of the distance window for pairing sites in the two-SFS. Default: 1000.")
    p.add_argument("--two-sfs-offset", dest="two_sfs_offset", type=int, default=0,
                   help="Minimum bp separation (exclusive) between paired sites for the two-SFS. Default: 0.")
    p.add_argument("--outgroups", type=_split_csv, default=None,
                   help="Outgroup samples (for the maximum-likelihood-ancestral annotation).")
    p.add_argument("--n-ingroups", dest="n_ingroups", type=_positive_int, default=11,
                   help="Minimum ingroups for the maximum-likelihood-ancestral annotation. Default: 11.")
    p.add_argument("--contigs", type=_split_csv, default=None,
                   help="Contigs to keep (for the contig filtration and stratification).")
    p.add_argument("--max-sites", type=_positive_int, default=None, help="Maximum number of sites to parse.")
    p.add_argument("--seed", type=int, default=0, help="Random seed. Default: 0.")
    p.set_defaults(handler=_run_parse)


def _add_filter_parser(sub: argparse._SubParsersAction) -> None:
    """
    Register the ``filter`` subcommand.

    :param sub: The subparsers action.
    """
    p = sub.add_parser("filter", help="Filter a dataset and write the result.",
                       description="Filter a VCF, VCF-Zarr store, or tree sequence using one or more filtrations. "
                                   "The output format follows the --output extension (a tree sequence may only be "
                                   "written from a tree-sequence input).")
    _add_common_io(p, "Output path; the format follows its extension (.vcf/.vcf.gz, .vcz/.zarr, or .trees).")

    p.add_argument("--filter", type=_split_csv, required=True,
                   help="Comma-separated filtrations (e.g. snp,coding-sequence,cpg).")
    p.add_argument("--fasta", default=None, help="FASTA reference (required by some filtrations, e.g. cpg).")
    p.add_argument("--gff", default=None, help="GFF annotation (required by the coding-sequence filtration).")
    p.add_argument("--contigs", type=_split_csv, default=None, help="Contigs to keep (for the contig filtration).")
    p.add_argument("--max-sites", type=_positive_int, default=None, help="Maximum number of sites to filter.")
    p.set_defaults(handler=_run_filter)


def _add_annotate_parser(sub: argparse._SubParsersAction) -> None:
    """
    Register the ``annotate`` subcommand.

    :param sub: The subparsers action.
    """
    p = sub.add_parser("annotate", help="Annotate a dataset and write the result.",
                       description="Annotate a VCF, VCF-Zarr store, or tree sequence with site degeneracy or "
                                   "ancestral-allele information. The output format follows the --output extension "
                                   "(a tree sequence may only be written from a tree-sequence input).")
    _add_common_io(p, "Output path; the format follows its extension (.vcf/.vcf.gz, .vcz/.zarr, or .trees).")

    p.add_argument("--annotation", type=_split_csv, required=True,
                   help="Comma-separated annotations (degeneracy, synonymy, maximum-likelihood-ancestral).")
    p.add_argument("--fasta", default=None, help="FASTA reference (required by the degeneracy annotation).")
    p.add_argument("--gff", default=None, help="GFF annotation (required by the degeneracy annotation).")
    p.add_argument("--outgroups", type=_split_csv, default=None,
                   help="Outgroup samples (for the maximum-likelihood-ancestral annotation).")
    p.add_argument("--n-ingroups", dest="n_ingroups", type=_positive_int, default=11,
                   help="Minimum ingroups for the maximum-likelihood-ancestral annotation. Default: 11.")
    p.add_argument("--info-ancestral", default="AA", help="INFO tag to write the ancestral allele to. Default: AA.")
    p.add_argument("--max-sites", type=_positive_int, default=None, help="Maximum number of sites to annotate.")
    p.add_argument("--seed", type=int, default=0, help="Random seed. Default: 0.")
    p.set_defaults(handler=_run_annotate)


def run(argv: Sequence[str] | None = None) -> int:
    """
    Parse arguments and dispatch to the selected subcommand.

    :param argv: Argument vector (defaults to ``sys.argv``).
    :return: Exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose, args.quiet)

    handler = getattr(args, "handler", None)

    if handler is None:
        parser.error("no subcommand selected")

    _check_output_distinct_from_input(args)

    return int(handler(args) or 0)


def main(argv: Sequence[str] | None = None) -> None:
    """
    Console-script entry point.

    :param argv: Argument vector (defaults to ``sys.argv``).
    """
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
