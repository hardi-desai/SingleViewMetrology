Following is an implementation of the paper “single view metrology”. It computes affine 3D geometry from a single perspective image using prior knowledge.

For annotation Line Segment Detection is used.

Post LSD, lines are thresholded using Length and Slope. It is observed that a single edge has multiple lines. These common lines are detected using the slope and one of the line is selected for calculation of vanishing point.

Reference Paper:
https://www.cs.cmu.edu/~ph/869/papers/Criminisi99.pdf
