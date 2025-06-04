import numpy as np
class T2NeutrosophicNumber:
    def __init__(self, truth, indeterminacy, falsity):
        self.truth = truth      
        self.indeterminacy = indeterminacy       
        self.falsity = falsity   

    def __repr__(self):
        return (f"t=({self.truth[0]:.3f}, {self.truth[1]:.3f}, "
        f"i={self.indeterminacy[0]:.3f}, {self.indeterminacy[1]:.3f}, "
        f"f={self.falsity[0]:.3f}, {self.falsity[1]:.3f})")

    def __add__(self, other):
     if isinstance(other.truth, (int, float, np.float64)):
        other_truth = (other.truth, other.truth, other.truth)
     else:
            other_truth = other.truth

     if isinstance(other.indeterminacy, (int, float, np.float64)):
        other_ind = (other.indeterminacy, other.indeterminacy, other.indeterminacy)
     else:
        other_ind = other.indeterminacy

     if isinstance(other.falsity, (int, float, np.float64)):
        other_falsity = (other.falsity, other.falsity, other.falsity)
     else:
        other_falsity = other.falsity

     truth = tuple(a + b - a * b for a, b in zip(self.truth, other_truth))
     indeterminacy = tuple(a * b for a, b in zip(self.indeterminacy, other_ind))
     falsity = tuple(a * b for a, b in zip(self.falsity, other_falsity))

     return T2NeutrosophicNumber(truth, indeterminacy, falsity)


    def __sub__(self, other):
         if isinstance(other.truth, (int, float, np.float64)):
            other_truth = (other.truth, other.truth, other.truth)
         else:
            other_truth = other.truth

         if isinstance(other.indeterminacy, (int, float, np.float64)):
            other_ind = (other.indeterminacy, other.indeterminacy, other.indeterminacy)
         else:
            other_ind = other.indeterminacy

         if isinstance(other.falsity, (int, float, np.float64)):
            other_falsity = (other.falsity, other.falsity, other.falsity)
         else:
            other_falsity = other.falsity

         truth = tuple(max(0, a - b) for a, b in zip(self.truth, other_truth))
         indeterminacy = tuple(max(0, a - b) for a, b in zip(self.indeterminacy, other_ind))
         falsity = tuple(max(0, a - b) for a, b in zip(self.falsity, other_falsity))

         return T2NeutrosophicNumber(truth, indeterminacy, falsity)


    def __mul__(self, other):
        truth = tuple(a * b for a, b in zip(self.truth, other.truth))
        indeterminacy = tuple(a + b - a * b for a, b in zip(self.indeterminacy, other.indeterminacy))
        falsity = tuple(a + b - a * b for a, b in zip(self.falsity, other.falsity))
        return T2NeutrosophicNumber(truth, indeterminacy, falsity)


    def score(self):
     alpha1, alpha2, alpha3 = list(self.truth) 
     beta1, beta2, beta3 = list(self.indeterminacy)
     gamma1, gamma2, gamma3 = list(self.falsity)

     return (1 / 12) * (8 + (alpha1 + 2 * alpha2 + alpha3) - (beta1 + 2 * beta2 + beta3) - (gamma1 + 2 * gamma2 + gamma3))




def classic_to_t2n(value, indeterminacy=0.1):
    truth = (value, value, value)
    falsity = (1 - value, 1 - value, 1 - value)
    indeterminacy = (indeterminacy, indeterminacy, indeterminacy)
    return T2NeutrosophicNumber(truth=truth, indeterminacy=indeterminacy, falsity=falsity)

