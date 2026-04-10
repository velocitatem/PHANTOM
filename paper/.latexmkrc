$pdf_mode = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 2;                       # run biber when biblatex .bcf changes
# biber cwd is paper/build; scripts/nx_paper.sh symlinks ../build/bib -> ../src/bib so
# datasources log as bib/references.bib and latexmk's -e check works from paper/src
$biber    = 'biber %O %S';

# Stale latexmk db: biblatex uses biber + .bcf, but the fdb can keep a "bibtex" rule after a bad
# run. Then biber never runs and citations stay undefined. Read whole fdb (small) so the rule
# line is never missed after a long dependency list.
for my $job (qw(main main-genpop summary)) {
    my $bcf = "../build/$job.bcf";
    my $bbl = "../build/$job.bbl";
    my $fdb = "../build/$job.fdb_latexmk";
    next unless -e $fdb && -e $bcf;
    my $drop = !-e $bbl;
    if ( !$drop && open my $fh, '<', $fdb ) {
        local $/;
        my $body = <$fh>;
        close $fh;
        $drop = 1 if defined $body && $body =~ /\["bibtex $job"\]/;
    }
    unlink $fdb if $drop;
}
$pdf_previewer = 'zathura %O %S';
$clean_ext = 'synctex.gz bbl bcf run.xml fls fdb_latexmk glg glo gls ist blg lof lot out toc';
