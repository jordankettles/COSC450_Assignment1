COSC450_Assignment1

# TODO
1. Implement DLT: Done.
2. Implement Normalised DLT: Done
3. Implement RANSAC without Normalisation: Done
Implement NormaliseValues function.
3. Implement RANSAC with Normalisation: 
4. Test how long variants of my method take to find a solution.
5. Test how many inlier correspondences are found in the RANSAC process with each variant.
6. Test how iterations does the RANSAC process need to find a good solution, and how many does this differ among the variants.
7. How accurately do the variants of my method estimate the homography?
The last point is key because to measure this I need to find or construct image pairs with a known homography. Then I will be able to determine the accuracy of my method. This comparison is complicated because homographies are only estimated up to scale.

I should calculate an error that is geometrically significant between the two homographies.
I should use multiple images.