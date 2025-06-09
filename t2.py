class T2NeutrosophicNumber:
    def __init__(self, truth, indeterminacy, falsity):
        self.truth = truth
        self.indeterminacy = indeterminacy
        self.falsity = falsity

    def __repr__(self):
        return (
            f"T={self.truth[0]:.3f}, {self.truth[1]:.3f}, {self.truth[2]:.3f}, "
            f"I={self.indeterminacy[0]:.3f}, {self.indeterminacy[1]:.3f}, {self.indeterminacy[2]:.3f}, "
            f"F={self.falsity[0]:.3f}, {self.falsity[1]:.3f}, {self.falsity[2]:.3f}"
        )


