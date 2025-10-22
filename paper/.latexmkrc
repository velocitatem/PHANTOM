$pdf_mode = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$aux_dir = 'build';
$out_dir = 'build';
$use_biber = 0;                        # force bibtex
$bibtex   = 'bibtex %O %B';
$pdf_previewer = 'zathura %O %S';
$clean_ext = 'synctex.gz bbl bcf run.xml fls fdb_latexmk glg glo gls ist blg lof lot out toc';
