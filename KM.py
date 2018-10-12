import numpy as np

class KM(object):
    def __init__(self, weight):
        self.weight = weight
        self.n = weight.shape[0]

    def dfs(self, x):
        self.visx[x] = True
        for y in range(self.n):
            if self.visy[y]:
                continue
            tmp = self.lx[x] + self.ly[y] - self.weight[x][y]
            if np.abs(tmp) < 1e-9:
                self.visy[y] = True
                if self.link[y] == -1 or self.dfs(self.link[y]):
                    self.link[y] = x
                    return True
            else:
                if self.slack[y] > tmp:
                    self.slack[y] = tmp
        return False

    def run(self):
        self.link = [-1] * self.n
        self.ly = [0] * self.n
        self.lx = [0] * self.n
        for i in range(self.n):
            self.lx[i] = -1e9
            for j in range(self.n):
                self.lx[i] = max(self.lx[i], self.weight[i, j])
        for x in range(self.n):
            self.slack = [1e9] * self.n
            while (True):
                self.visx = [False] * self.n
                self.visy = [False] * self.n
                if self.dfs(x):
                    break
                d = 1e9
                for i in range(self.n):
                    if (not self.visy[i]) and (d > self.slack[i]):
                        d = self.slack[i]
                for i in range(self.n):
                    if self.visx[i]:
                        self.lx[i] -= d
                for i in range(self.n):
                    if self.visy[i]:
                        self.ly[i] += d
                    else:
                        self.slack[i] -= d
        res = 0
        for i in range(self.n):
            if self.link[i] != -1:
                res += self.weight[self.link[i]][i]
        return res

def min_KM(weight):
    a = KM(-weight)
    res = -a.run()
    cor = a.link
    return res, cor

if __name__ == '__main__':
    import numpy as np
    w = np.array([[12., 6.], [5., 0.]])
    print(min_KM(w))

