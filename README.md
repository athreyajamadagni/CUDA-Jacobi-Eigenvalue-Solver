# CUDA-Jacobi-Eigenvalue-Solver

CUDA Jacobi Eigenvalue Solver

Developed a CUDA implementation of the Jacobi eigenvalue algorithm to solve the eigenvalue problem for symmetric matrices. The implementation is optimized for performance and scalability, and it can be used to solve large-scale eigenvalue problems on GPUs.

AIM:
To find the Eigen Values of A matrix using CUDA

METHODOLOGY:
In this project, we use the Jacobi eigen Value Algorithm to find all the eigen values of a matrix.


Jacobi Algorithm
The Jacobi eigenvalue algorithm is a numerical method for computing the eigenvalues and eigenvectors of a real symmetric matrix. It is a well-known iterative algorithm that can be used to find all eigenvalues and eigenvectors of a symmetric matrix. The basic idea of the algorithm is to repeatedly perform a similarity transformation on the matrix until the off-diagonal elements become small enough to be considered zero. This transformation is accomplished by using a sequence of rotations that preserve the symmetry of the matrix.

The Jacobi algorithm works by iteratively applying a series of orthogonal transformations to a symmetric matrix. Each transformation rotates the matrix in a way that eliminates one of the off-diagonal elements, and the process is repeated until all of the off-diagonal elements are zero. At each step, the algorithm selects the off-diagonal element with the largest absolute value and performs a rotation to make it zero. This process is repeated until all the off-diagonal elements are zero. The resulting diagonal elements are the eigenvalues of the matrix.



ALGORITHM STEPS:

1.	Start with a symmetric matrix A.
2.	Choose an initial matrix P, which is usually taken to be the identity matrix.
3.	While the off-diagonal elements of A are not sufficiently small (i.e., while the matrix is not yet diagonal), do the following:
a.	Find the largest off-diagonal element of A, call it a_ij.
b.	Compute the rotation angle theta = 0.5 * atan(2*a_ij/(a_ii - a_jj)).
c.	Construct the rotation matrix R(theta) by setting R_ii = R_jj = cos(theta), R_ij = -sin(theta), R_ji = sin(theta), and all other elements of R to zero.
d.	Update the matrix A by computing A = R^T * A * R.
e.	Update the matrix P by computing P = P * R.
4.	Once the off-diagonal elements of A are sufficiently small (e.g., less than some tolerance value), the diagonal elements of A are the eigenvalues of the original matrix A, and the columns of P are the corresponding eigenvectors.


CUDA Implementation Block Diagram:

![image](https://github.com/athreyajamadagni/CUDA-Jacobi-Eigenvalue-Solver/assets/75878205/1686c04a-2761-4b39-8388-71fad298ceb3)

Functions being used in the code:

1.	r8mat_diag_get_vector: given a square symmetric matrix a of size n and an array v of size n, this function extracts the diagonal entries of a and stores them in v.

2.	 r8mat_identity: given an array a of size n^2, this function sets a to the identity matrix of size n.

3.	 jacobi_eigenvalue: given a square symmetric matrix a of size n, an integer it_max specifying the maximum number of iterations to perform, arrays v and d of size n^2 and n, respectively, and integers it_num and rot_num to store the number of iterations and rotations performed, respectively, this function diagonalizes a using the Jacobi eigenvalue algorithm. The eigenvectors are stored in v and the eigenvalues are stored in d

Cuda functions used:
1.	cudaMalloc: This function is used to allocate memory on the GPU.

2.	cudaMemcpy: This function is used to transfer data between the host (CPU) and the device (GPU).

3.	cudaFree: This function is used to free memory that was previously allocated on the GPU.

4.	cudaDeviceSynchronize: This function is used to synchronize the host and device threads. It ensures that all previously issued CUDA commands have completed.

5.	<<<...>>>: This is a CUDA kernel launch configuration. It specifies the number of blocks and threads to launch on the GPU.

6.	_ _global_ _: This is a CUDA keyword used to indicate a function that will run on the GPU.

7.	_ _device_ _: This is a CUDA keyword used to indicate a function that will run on the GPU and be called from other device functions.

8.	_ _shared_ _: This is a CUDA keyword used to indicate that a variable should be stored in shared memory on the GPU. Shared memory is faster than global memory but has limited capacity.


How the Value is copied between the Host (CPU) and The Device (GPU):

1.	The variables h_a, h_v, and h_d are pointers to double precision arrays allocated on the host using malloc(). The variables d_a, d_v, and d_d are pointers to double precision arrays allocated on the device using cudaMalloc().

2.	The memset() function sets all the elements of h_a, h_v, and h_d to zero.

3.	The initialize_matrix() function initializes h_a with input matrices a1, a2, and a3.

4.	The cudaMemcpy() function is used to copy the contents of the arrays h_a, h_v, and h_d from the host to the device.

5.	The kernel function je is then launched on the device using the <<<>>> syntax with appropriate parameters.

6.	After the kernel completes execution, the output d_d is copied back to the host using cudaMemcpy() and the print_vec() function is called to print the output matrices d for each input matrix a.

7.	Finally, the memory allocated on the host is freed using free() and the memory allocated on the device is freed using cudaFree().



OBSERVATION:

This Jacobi eigenvalue algorithm implementation in C++ calculates the eigenvalues and eigenvectors of a real symmetric matrix. To diagonalize the matrix, the algorithm rotates the matrix repeatedly. The implementation comes with a number of helpful functions, including ones that can extract a matrix's diagonal, build an identity matrix, and determine the norm of the rigorous upper triangle of a matrix.

The maximum number of iterations, a N*N square symmetric matrix, and two arrays to hold the matrix's eigenvectors and eigenvalues are all inputs required by the method. The implementation converts the input eigenvector matrix to an orthogonal matrix that diagonalizes it, and then saves the eigenvalues in an input eigenvalue array.

The implementation initialises two arrays to hold the diagonal and off-diagonal matrix members, as well as an iteration counter, a rotation counter, and a rotation counter. The rotation counter counts the total number of rotations, whereas the iteration counter regulates the maximum number of iterations.


When the maximum number of iterations has been reached or the norm of the stringent upper triangle of the matrix is below a predetermined value, the implementation exits the loop. The loop iterates over all of the matrix's off-diagonal entries and rotates each one that is greater than the cutoff value. The off-diagonal element will be destroyed by the rotation by becoming zero, while the matrix's diagonal members will be preserved. The method employs a number of techniques to make the rotations efficient and stable, such as selecting the rotation angle to reduce the norm of the updated matrix's stringent upper triangle.
