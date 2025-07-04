import math
import operator
import random
from decimal import Decimal


class Matrix:

    def __init__(self, m = 0, n = 0, data = None):
        """
        :param m: number of rows
        :param n: number of columns
        :param data: matrix data (if it exists)
        """

        self.m = len(data) if data else m
        self.n = len(data[0]) if data else n
        self.matrix = [[float(val) for val in row] for row in data] if data \
            else [[0 for _ in range(n)] for _ in range(m)]


    def randomize(self):
        """
        For each value in the matrix, turns that value into a random value between -1.0 and 1.0
        :return: None
        """
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                self.matrix[i][j] = random.randint(-50, 50)/50


    def T(self):
        """
        Returns the transposition of the matrix
        :return: a matrix form of {self.matrix}'s transposition
        """
        return Matrix(data=list(map(list, zip(*self.matrix))))

    def dot(self, other):
        """
        Returns the dot product of this matrix and another matrix
        Dimensions of new matrix are: (self.m, other.n)

        :param other: other matrix
        :return: the new matrix
        """
        if type(other) is not Matrix:
            raise TypeError("Cannot do dot product with non-Matrix type:", type(other))
        if self.n != other.m:
            raise IndexError(f"Dot product: Matrix sizes are incompatible ({self.m}, {self.n}), and ({other.m}, {other.n})")


        new_matrix = Matrix(self.m, other.n)     # Resulting matrix self rows, and other cols

        for i in range(self.m):
            for j in range(other.n):
                # running algorithm to place at i, j
                for k in range(self.n):
                    new_matrix.matrix[i][j] += self.matrix[i][k] * other.matrix[k][j]
        return new_matrix

    def get_column(self, col_index):
        """
        :param col_index: column index
        :return: the column at the given index
        """
        return [self.matrix[i][col_index] for i in range(self.m)]

    def softmax(self):
        """
        Applies softmax across each column (i.e., across class scores).
        For a matrix of shape (num_classes, batch_size), applies softmax over axis=0.
        """
        new_matrix = Matrix(self.m, self.n)  # same shape

        for j in range(self.n):  # for each column (sample)
            column = self.get_column(j)
            exp_vals = [math.exp(val) for val in column]
            total = sum(exp_vals)
            for i in range(self.m):
                new_matrix.matrix[i][j] = exp_vals[i] / total

        return new_matrix

    def ReLU(self):
        """
        Creates a new matrix copy of {self.matrix} such that all the negative values in the matrix are 0.
        :return: the ReLU matrix
        """
        new_matrix = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                new_matrix.matrix[i][j] = max(0, self.matrix[i][j])
        return new_matrix

    def d_ReLU(self):
        new_matrix = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                new_matrix.matrix[i][j] = 1 if self.matrix[i][j] > 0 else 0
        return new_matrix

    def get_matrix_averaged_column(self):
        new_matrix = Matrix(self.m, 1)
        for i, row in enumerate(self.matrix):
            new_matrix.matrix[i][0] = sum(row)/len(self.matrix[i])
        return new_matrix

    def get_matrix_maxes(self):
        max_indices = []
        for j in range(self.n):  # for each column
            max_val = self.matrix[0][j]
            max_row = 0
            for i in range(1, self.m):
                if self.matrix[i][j] > max_val:
                    max_val = self.matrix[i][j]
                    max_row = i
            max_indices.append(max_row)
        return max_indices

    def __getitem__(self, item):
        """
        Gets the value of {self.matrix[r][c]}
        Used like: matrix[1, 5]
        :param item: row index, col index
        :return: {self.matrix[r][c]}
        """
        r, c = item
        return self.matrix[r][c]

    def __setitem__(self, item, value):
        """
        Sets the value of {self.matrix[r][c]}
        :param item: row index, col index
        :param value: the new value
        :return: None
        """
        r, c = item
        self.matrix[r][c] = value


    def __truediv__(self, m):
        """
        Only handles scalar division
        :param m: a scalar value
        :return: a new matrix with the scalar multiplication applied
        """

        new_matrix = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                new_matrix.matrix[i][j] = self.matrix[i][j]/m
        return new_matrix


    def __mul__(self, other):
        """
        Handles scalar multiplication and the hadamard product, adapting automatically based on
        {type.other}
        :param other: either a matrix or a scalar value
        :return: a new matrix with the multiplication applied
        """
        if type(other) is int or type(other) is float:
            # Scalar multiplication
            return self.__truediv__(1/other)

        # Hadamard product
        if self.n != other.n or self.m != other.m:
            raise IndexError(f"Hadamard product: Index error matrix sizes ({self.m}, {self.n}), and ({other.m}, {other.n}) must be the same.")

        new_matrix = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                new_matrix.matrix[i][j] = self.matrix[i][j] * other.matrix[i][j]

        return new_matrix


    def get_add_sub_func(self, other, oper):
        if self.n == other.n and self.m == other.m:
            # No broadcasting, default matrix addition/subtraction
            get_val = lambda i, j: oper(self.matrix[i][j], other.matrix[i][j])
        elif self.n == other.n and self.m == 1:
            get_val = lambda i, j: oper(self.matrix[0][j], other.matrix[i][j])
        elif self.m == other.m and self.n == 1:
            get_val = lambda i, j: oper(self.matrix[i][0], other.matrix[i][j])
        elif self.n == other.n and other.m == 1:
            get_val = lambda i, j: oper(self.matrix[i][j], other.matrix[0][j])
        elif self.m == other.m and other.n == 1:
            get_val = lambda i, j: oper(self.matrix[i][j], other.matrix[i][0])
        else:
            raise IndexError(f"Addition: matrix sizes are incompatible ({self.m}, {self.n}), and ({other.m}, {other.n}).")
        return get_val

    def __sub__(self, other):
        """
        Subtracts two matrices from each other. The function does not mess with any of the values in {self} or {other}.
        Unlike traditional matrix addition, this function allows for 'broadcasting' (ie: (a, b) - (a, 1)).
        :param other: another matrix
        :return: the new matrix, that represents the sum of the two matrices
        """
        get_val = self.get_add_sub_func(other, operator.sub)
        # Result shape always matches the larger shape
        new_matrix = Matrix(max(self.m, other.m), max(self.n, other.n))
        for i in range(new_matrix.m):
            for j in range(new_matrix.n):
                new_matrix.matrix[i][j] = get_val(i, j)

        return new_matrix


    def __add__(self, other):
        """
        Adds two matrices together. The function does not mess with any of the values in {self} or {other}.
        Unlike traditional matrix addition, this function allows for 'broadcasting' (ie: (a, b) + (a, 1)).
        :param other: another matrix
        :return: the new matrix, that represents the sum of the two matrices
        """

        get_val = self.get_add_sub_func(other, operator.add)
        # Result shape always matches the larger shape
        new_matrix = Matrix(max(self.m, other.m), max(self.n, other.n))
        for i in range(new_matrix.m):
            for j in range(new_matrix.n):
                new_matrix.matrix[i][j] = get_val(i, j)
        return new_matrix


    def __str__(self):
        max_rows = min(self.m, 8)
        max_cols = min(self.n, 8)
        s = f"\n{5*'\t'}Matrix shape: ({self.m}, {self.n})\n"

        for i in range(max_rows):
            row_str = "\t"
            for j in range(max_cols):
                val = self.matrix[i][j]
                row_str += '%.1E' % Decimal(val) + ' '  # 2 significant figures
            if self.n > 8:
                row_str += "... "
            s += row_str.rstrip() + "\t\n"

        if self.m > 8:
            s += "\t.\n\t.\n\t."

        return s
