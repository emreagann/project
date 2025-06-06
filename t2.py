class T2NeutrosophicNumber:
    def __init__(self, truth, indeterminacy, falsity):
        self.truth = truth             
        self.indeterminacy = indeterminacy  
        self.falsity = falsity          

    def __repr__(self):
        return (
            f"T={self.truth[0]:.3f}, {self.truth[1]:.3f}, {self.truth[2]:.3f}, "
            f"I={self.indeterminacy[0]:.3f}, {self.indeterminacy[1]:.3f}, {self.indeterminacy[2]:.3f}, "
            f"F={self.falsity[0]:.3f}, {self.falsity[1]:.3f}, {self.falsity[2]:.3f})"
        )

    def __add__(self, other):
        truth = tuple(a + b - a * b for a, b in zip(self.truth, other.truth))
        indeterminacy = tuple(a * b for a, b in zip(self.indeterminacy, other.indeterminacy))
        falsity = tuple(a * b for a, b in zip(self.falsity, other.falsity))
        return T2NeutrosophicNumber(truth, indeterminacy, falsity)

    def __sub__(self, other):
        truth = tuple(max(0, a - b) for a, b in zip(self.truth, other.truth))
        indeterminacy = tuple(max(0, a - b) for a, b in zip(self.indeterminacy, other.indeterminacy))
        falsity = tuple(max(0, a - b) for a, b in zip(self.falsity, other.falsity))
        return T2NeutrosophicNumber(truth, indeterminacy, falsity)

    def __mul__(self, other):
        truth = tuple(a * b for a, b in zip(self.truth, other.truth))
        indeterminacy = tuple(a + b - a * b for a, b in zip(self.indeterminacy, other.indeterminacy))
        falsity = tuple(a + b - a * b for a, b in zip(self.falsity, other.falsity))
        return T2NeutrosophicNumber(truth, indeterminacy, falsity)

    def score(self):
        t1, t2, t3 = self.truth
        i1, i2, i3 = self.indeterminacy
        f1, f2, f3 = self.falsity
        return (1 / 12) * (8 + (t1 + 2*t2 + t3) - (i1 + 2*i2 + i3) - (f1 + 2*f2 + f3))

    def __le__(self, other):
        return self.score() <= other.score()

    def __lt__(self, other):
        return self.score() < other.score()

    def __ge__(self, other):
        return self.score() >= other.score()

    def __gt__(self, other):
        return self.score() > other.score()