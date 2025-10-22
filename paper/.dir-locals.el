((latex-mode
  . ((TeX-engine . default) ; or luatex/xetex
     (TeX-command-extra-options . "-file-line-error -interaction=nonstopmode")
     (TeX-master . "/home/velocitatem/Documents/Projects/PHANTOM/paper/src/main.tex")
     (TeX-source-correlate-mode . t)
     (TeX-source-correlate-start-server . t)
     (TeX-PDF-mode . t)
     (reftex-mode . t)
     (reftex-cite-format . "\\parencite{%l}")
     (fill-column . 100))))
