$pdf_mode = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 2;                       # run biber when biblatex .bcf changes
# biber runs with cwd = -outdir (paper/build); .bib paths are relative to paper/src
$biber    = 'biber --input-directory=../src %O %S';
$pdf_previewer = 'zathura %O %S';
$clean_ext = 'synctex.gz bbl bcf run.xml fls fdb_latexmk glg glo gls ist blg lof lot out toc';
