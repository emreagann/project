class T2NeutrosophicNumber:
    def __init__(self, truth, indeterminacy, falsity):
        self.truth = truth      
        self.indeterminacy = indeterminacy       
        self.falsity = falsity   

    def __repr__(self):
        return f"T2N(truth={self.truth:.3f}, indeterminacy={self.indeterminacy:.3f}, falsity={self.falsity:.3f})"

    def __add__(self, other):
     return T2NeutrosophicNumber(
        self.truth + other.truth,
        self.indeterminacy + other.indeterminacy,
        self.falsity + other.falsity
     )


    def __mul__(self, other):
        return T2NeutrosophicNumber(
            self.truth * other.truth,
            self.indeterminacy + other.indeterminacy - self.indeterminacy * other.indeterminacy,
            self.falsity + other.falsity - self.falsity * other.falsity
        )

    def score(self):
  
     val = (8 + (self.truth**2 + 2*self.truth*self.indeterminacy + self.truth*self.falsity)
             - (self.indeterminacy*self.truth + 2*self.indeterminacy**2 + self.indeterminacy*self.falsity)
             - (self.falsity*self.truth + 2*self.falsity*self.indeterminacy + self.falsity**2))

     return val / 12


def classic_to_t2n(value, indeterminacy=0.1):

   
    return T2NeutrosophicNumber(truth=value, indeterminacy=indeterminacy, falsity=1-value)
