$pdf_mode = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 2;                       # run biber when biblatex .bcf changes
# biber cwd is paper/build; scripts/nx_paper.sh symlinks ../build/bib -> ../src/bib so
# datasources log as bib/references.bib and latexmk's -e check works from paper/src
$biber    = 'biber %O %S';
$pdf_previewer = 'zathura %O %S';
$clean_ext = 'synctex.gz bbl bcf run.xml fls fdb_latexmk glg glo gls ist blg lof lot out toc';
