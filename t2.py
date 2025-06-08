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
def t2nn_score(t2nn):
    if isinstance(t2nn, T2NeutrosophicNumber):
        t = t2nn.truth
        i = t2nn.indeterminacy
        f = t2nn.falsity
        score = (8 + (t[0] + 2*t[1] + t[2]) - (i[0] + 2*i[1] + i[2]) - (f[0] + 2*f[1] + f[2])) / 12
        return score
    elif isinstance(t2nn, (int, float)):
        return float(t2nn)
    else:
        return 0.0
