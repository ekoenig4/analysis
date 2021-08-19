# analysis
Framework for analysis of events with 6 bs in the final state.

## Input
Input files are generated using the [sixB analyzer](https://github.com/ekoenig4/sixB/tree/master/analysis/sixBanalysis)

## Structure
Jupyter notebook studies are saved in the jupyter directory, each with a similar init cell. The utils package has the ability to read TTrees from the sixB analyzer using the Tree class. The Selection class can be used to make more cuts and filters on the Tree class. A jupyter study consists of calls to the studyUtils package to plot certain variables. A handful of plots have been hardcoded for easy access, but the study.quick method can be used to plot any variable stored in the Tree class.

## Plots
Plots that are saved when using the saveas variable for the study methods are saved to a dated plots directory in the git base.
