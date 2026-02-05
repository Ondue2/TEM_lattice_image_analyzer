## TEM_lattice_image_analyzer

This code is designed to extract atomic coordinates and intensities from TEM images of lattice structures.

This code can analyze thousands of atoms, or even tens of thousands, simultaneously, enabling statistical analysis.

The analysis result is useful for capturing subtle symmetry-breaking phenomena, such as slight atomic displacements that break the original symmetry, vacancy orderings, and stacking disorders.

This code uses two features for the analysis: peak finding using `skimage.feature.peak_local_max` and the gradient descent method in `TensorFlow`, minimizing the MSE between the experimental TEM image and a simulated TEM image constructed from ellipsoidal Gaussian functions.

You can use a combination of peak finding and Gaussian fitting effectively, which might allow analysis of subtle cases: weak atomic signals, spatially overlapping atomic signals, etc.

The following images are the experimental TEM image and the simulated TEM image after the optimization process.

<img width="844" height="452" alt="image" src="https://github.com/user-attachments/assets/b47c4931-8b2b-4d23-aded-e9171071aef2" />

One useful analysis from this optimization is the histogram of Gaussian intensities, as follows.

<img width="593" height="467" alt="image" src="https://github.com/user-attachments/assets/fce58754-c5f7-4d37-af18-867fdb48df0d" />

This is very useful to investigate atomic occupancies. If the distribution of the histogram of a specific atom is broad, the atom might have a disordered occupation. Or you can compare atomic occupancies between symmetrically equivalent atoms, so you might find a hidden symmetry breaking from the vacancy ordering.

Another useful application is to investigate atomic displacement from the symmetric center, as follows.

<img width="605" height="470" alt="image" src="https://github.com/user-attachments/assets/b80427b6-c8ca-4469-8a83-301d4fe43734" />


A research using this analysis can be found in [here](https://arxiv.org/abs/2507.23068).


Please see [here]() for the detailed workflow instruction.






