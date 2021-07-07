class MulLayer: # 乗算レイヤー
    def __init__(self):
        self.x = None
        self.y = None

    # 順伝播の計算
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    # 逆伝播の計算
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer: # 加算レイヤー
    def __init__(self):
        pass

    # 順伝播の計算
    def forward(self, x, y):
        out = x + y
        return out

    # 逆伝播の計算
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
