def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
	eigenvalues = []
	eigenvalues.append(((a + d) + ((a + d) ** 2 - 4 * a * d + 4 * b * c) ** 0.5) / 2)
	eigenvalues.append(((a + d) - ((a + d) ** 2 - 4 * a * d + 4 * b * c) ** 0.5) / 2)
	return sorted(eigenvalues, reverse=True)