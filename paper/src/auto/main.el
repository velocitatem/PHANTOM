;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "main"
 (lambda ()
   (setq TeX-command-extra-options
         "-file-line-error -interaction=nonstopmode")
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("report" "12pt") ("article" "12pt") ("acmart" "sigconf" "nonacm" "natbib=false")))
   (TeX-run-style-hooks
    "latex2e"
    "preamble"
    "acmart"
    "acmart10")
   (TeX-add-symbols
    '("footnotetextcopyrightpermission" 1))
   (LaTeX-add-labels
    "research"))
 :latex)

