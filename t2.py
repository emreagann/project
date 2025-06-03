class T2NeutrosophicNumber:
    def __init__(self, truth, indeterminacy, falsity):
        self.truth = truth      
        self.indeterminacy = indeterminacy       
        self.falsity = falsity   

    def __repr__(self):
        return f"T2N(truth={self.truth:.3f}, indeterminacy={self.indeterminacy:.3f}, falsity={self.falsity:.3f})"

    def __add__(self, other):
        return T2NeutrosophicNumber(
            self.truth + other.truth - self.truth * other.truth,
            self.indeterminacy * other.indeterminacy,
            self.falsity * other.falsity
        )

    def __sub__(self, other):
        return T2NeutrosophicNumber(
            max(0, self.truth - other.truth),
            max(0, self.indeterminacy - other.indeterminacy),
            max(0, self.falsity - other.falsity)
        )

    def __mul__(self, other):
        return T2NeutrosophicNumber(
            self.truth * other.truth,
            self.indeterminacy + other.indeterminacy - self.indeterminacy * other.indeterminacy,
            self.falsity + other.falsity - self.falsity * other.falsity
        )

    def score(self):
        return self.truth - self.falsity


def classic_to_t2n(value, indeterminacy=0.1):
    truth = min(max(value, 0), 1)
    falsity = min(max(1 - value, 0), 1)
    return T2NeutrosophicNumber(truth=truth, indeterminacy=indeterminacy, falsity=falsity)
