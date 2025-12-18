# NeutrophilShapeAnalysis
This repository functions to reproduce the results published in ______.

## Workflow
All of the image and data processing work are performed in jupyter notebooks found in 'Notebooks/' and are run in the following order:
1. Segment_and_Track_Motility_Paper_Confocal_Data.ipynb / Segment_and_Track_Motility_Paper_LLS_Random_Only.ipynb
	- These notebooks segment, track, and crop individual cells from larger images.
	- This step is separate from other data processing steps because it is unlikely to change regardless of other decisions about the analysis process.
2. Processing_Motility_Paper_Confocal_Data.ipynb / Processing_Motility_Paper_LLS_Random_Only.ipynb
	- These notebooks find the alignment vectors to put cells in a common frame of reference based on their trajectory, align cell meshes converted from segmented images, and calculate Spherical Harmonic (SH) coefficients.
3. PCA_with_all_37C_confocal_data.ipynb / PCA_with_all_37C_confocal_data_LLS_apply.ipynb
	- These notebooks perform additional quality control steps and then us PCA on SH coefficients from all confocal data and apply the same PCA transform to LLS data.
4. Processing_Motility_Paper_Confocal_Detailed_Balance.ipynb / Processing_Motility_Paper_LLS_Detailed_Balance.ipynb
	- These notebooks use the PCs calculated in the previous step to calculate Coarse Grained Phase Spaces (CGPSs) for pairs of top PCs and Area Enclosing Rates for real cell trajectories and bootstrapped trajectories in those spaces.

## Figures and Data
