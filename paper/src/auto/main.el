;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "main"
 (lambda ()
   (setq TeX-command-extra-options
         "-file-line-error -interaction=nonstopmode")
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("report" "12pt") ("article" "12pt") ("acmart" "sigconf" "nonacm" "natbib=false" "manuscript")))
   (TeX-run-style-hooks
    "latex2e"
    "preamble"
    "chapters/01-intro"
    "chapters/02-literature-review"
    "chapters/03-methodology"
    "chapters/04-results"
    "chapters/05-discussion"
    "chapters/06-conclusion"
    "../build/concatenated_code"
    "acmart"
    "acmart10")
   (TeX-add-symbols
    '("footnotetextcopyrightpermission" 1)))
 :latex)

