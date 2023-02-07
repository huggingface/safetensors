import paddle

weights = paddle.load("paddle_ace.pdparams")[0]
assert list(weights.keys()) == ["weight"]
assert paddle.allclose(weights["weight"], paddle.zeros((2, 2)))
print("The file looks fine !")
