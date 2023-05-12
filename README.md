# Big NNLMs and De Re / De Dicto Distinction
A repository for a project investigating NNLM sensitivity to the De Re / De Dicto distinction. Most code used for the project is in this repository; code for the statistical analysis we conduct, along with descriptions of the repository files, will go up shortly.

The `statsandplots` folder contains the plots and regression included in the paper text, as well as the R script used to generate them. Running the R script from start to finish with `statsandplots` as the working directory will re-generate all of the plots, and will also re-fit the mixed-effects linear regression we report in the text. The regression takes several hours to fit, over 12 hours on one author's personal computer. To skip this step and read our regression from saved parameters, there are particular lines marked in the R file that should be commented out or deleted. If this is done, the R script should run from start to finish within a few minutes. Note that `git lfs` is needed to clone or download the `.rds` file containing the saved regression parameters.
