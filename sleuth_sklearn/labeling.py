def hoshen_kopelman(matrix):
    matrix = matrix.copy()

    m, n = matrix.shape
    max_labels = (m * n) // 2

    labels = np.empty(max_labels, dtype=int)
    labels[0] = 0

    def find(x):
        y = x
        while labels[y] != y:
            y = labels[y]

        while labels[x] != x:
            z = labels[x]
            labels[x] = y
            x = z
        return y

    def union(x, y):
        val = find(y)
        labels[find(x)] = val
        return val

    def make_set():
        labels[0] += 1
        labels[labels[0]] = labels[0]
        return labels[0]

    for i in range(m):
        for j in range(n):
            if matrix[i][j] > 0:
                up = matrix[i - 1][j] if i > 0 else 0
                left = matrix[i][j - 1] if j > 0 else 0
                if up == 0 and left == 0:
                    matrix[i][j] = make_set()
                elif up == 0 or left == 0:
                    matrix[i][j] = max(up, left)
                else:
                    matrix[i][j] = union(up, left)

    new_labels = np.zeros(m * n // 2, dtype=int)
    for i in range(m):
        for j in range(n):
            if matrix[i, j] > 0:
                x = find(matrix[i, j])
                if new_labels[x] == 0:
                    new_labels[0] += 1
                    new_labels[x] = new_labels[0]
                matrix[i, j] = new_labels[x]

    return matrix, new_labels[0]
