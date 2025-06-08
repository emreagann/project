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

def normalize_t2nn(value, min_val, max_val, ctype):
    if isinstance(value, T2NeutrosophicNumber):
        def norm(x, min_x, max_x, is_benefit):
            if max_x - min_x == 0:
                return 0.0
            return (x - min_x) / (max_x - min_x) if is_benefit else (max_x - x) / (max_x - min_x)

        t = [norm(value.truth[i], min_val.truth[i], max_val.truth[i], ctype == "benefit") for i in range(3)]
        i = [norm(value.indeterminacy[i], min_val.indeterminacy[i], max_val.indeterminacy[i], ctype == "cost") for i in range(3)]
        f = [norm(value.falsity[i], min_val.falsity[i], max_val.falsity[i], ctype == "cost") for i in range(3)]
        return T2NeutrosophicNumber(tuple(t), tuple(i), tuple(f))
    
    elif isinstance(value, (int, float)):
        if max_val - min_val == 0:
            return 0.0
        return (value - min_val) / (max_val - min_val) if ctype == "benefit" else (max_val - value) / (max_val - min_val)

    return 0.0

