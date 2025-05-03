# Introduction

Risk measurement relies on modelling assumptions. Errors in these assumptions introduce errors in risk measurement. This makes risk measurement vulnerable to model risk.
This project develops tools for quantifying model risk and making risk measurement robust to modelling errors. One core aspect of this paper is to go beyond distribution's parameter errors and find the true worst case scenarios.

The paper [Robust Risk Measurement and Model Risk](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2167765) was published by PAUL GLASSERMAN and XINGBO XU in 2012.

This project contains the Robust risk measurement over permissible relative Entropy in different risk measurement scenarios.
The relative Entropy is rather a calculated quantity which shows the permissible model deviation which is ultimately reached in the worst case scenario. The paper builds this idea in a sequential manner in various sections. The mathematical derivations of formulas and results are presented in the pdf file- [here](Derivations_Conclusions.pdf)

# Contents

This project contains the code implementation of the quantitative risk analysis done in the paper.
The root directory contains 3 files-
MeanVarianceOpt.ipynb - This contains the code implementation of the analysis in section 4.1. 
ConditionalVaRisk.ipynb - This python notebook has code implementation of the analysis in section 5.1 & 5.2. 
GaussianCopula.ipynb - The section contains the code for robustness to the problem of portfolio credit risk measurement.

The folder 'MeanVarianceImperical' contains the real examples of portfolio of stock and focusses on predicting the worst case risk in stressful scenarios.

Note:- Structurization of this project is still incomplete and will be published in the future.
