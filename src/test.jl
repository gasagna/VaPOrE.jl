begin
    # global row indices of the i-th block
    _blockrng(i::Integer, n::Integer) = ((i-1)*n+1):(i*n)


    # size
    M = 100
    N = 3

    # allocate
    A = spzeros(M*N+2, M*N+2)

    # main diagonal
    for i = 1:M
        A[_blockrng(i, N), _blockrng(i, N)] .= rand.()
    end

    # off diagonal
    A[diagind(A, +N)] =  1
    A[diagind(A, -N)] = -1

    # off diagonal
    A[_blockrng(1, N), _blockrng(M, N)] = -eye(3)
    A[_blockrng(M, N), _blockrng(1, N)] =  eye(3)

    # borders
    A[:, end-1:end] = rand.()
    A[end-1:end, :] = rand.()
    A[end-1:end, end-1:end] = 0
end

using PyPlot; pygui(true)
matshow(A)
show()
