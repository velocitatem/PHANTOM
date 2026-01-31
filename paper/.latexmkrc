$pdf_mode = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 2;                       # run bibtex when needed
$bibtex   = 'bibtex %O %B';
$pdf_previewer = 'zathura %O %S';
$clean_ext = 'synctex.gz bbl bcf run.xml fls fdb_latexmk glg glo gls ist blg lof lot out toc';
