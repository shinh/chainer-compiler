import torch
import torch.nn as nn
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward
from chainer_compiler.elichika.testtools import type_inference_tools

from testcases.pytorch.vae               import gen_VAE_model


class TestVAE(unittest.TestCase):
    def test_VAE(self):
        type_inference_tools.reset_state()

        model, forward_args = gen_VAE_model()
        id2type = generate_id2type_from_forward(model, forward_args)

        # === BEGIN ASSERTIONS for VAE ===
        # === function forward ===
        self.assertEqual(str(id2type[8]), "(torch.Tensor(float32, (128, 20)), torch.Tensor(float32, (128, 20)))")	# Tuple (mu, logvar) (line 2)
        self.assertEqual(str(id2type[9]), "torch.Tensor(float32, (128, 20))")	# Name mu (line 2)
        self.assertEqual(str(id2type[11]), "torch.Tensor(float32, (128, 20))")	# Name logvar (line 2)
        self.assertEqual(str(id2type[14]), "(torch.Tensor(float32, (128, 20)), torch.Tensor(float32, (128, 20)))")	# Call self.encode(x.view(-1, 784)) (line 2)
        self.assertEqual(str(id2type[16]), "class VAE")	# Name self (line 2)
        self.assertEqual(str(id2type[19]), "torch.Tensor(float32, (128, 784))")	# Call x.view(-1, 784) (line 2)
        self.assertEqual(str(id2type[21]), "torch.Tensor(float32, (128, 1, 28, 28))")	# Name x (line 2)
        self.assertEqual(str(id2type[24]), "int")	# UnaryOp -1 (line 2)
        self.assertEqual(str(id2type[26]), "int")	# Constant 1 (line 2)
        self.assertEqual(str(id2type[27]), "int")	# Constant 784 (line 2)
        self.assertEqual(str(id2type[29]), "torch.Tensor(float32, (128, 20))")	# Name z (line 3)
        self.assertEqual(str(id2type[31]), "torch.Tensor(float32, (128, 20))")	# Call self.reparameterize(mu, logvar) (line 3)
        self.assertEqual(str(id2type[33]), "class VAE")	# Name self (line 3)
        self.assertEqual(str(id2type[36]), "torch.Tensor(float32, (128, 20))")	# Name mu (line 3)
        self.assertEqual(str(id2type[38]), "torch.Tensor(float32, (128, 20))")	# Name logvar (line 3)
        self.assertEqual(str(id2type[41]), "(torch.Tensor(float32, (128, 784)), torch.Tensor(float32, (128, 20)), torch.Tensor(float32, (128, 20)))")	# Tuple (self.decode(z), mu, logvar) (line 4)
        self.assertEqual(str(id2type[42]), "torch.Tensor(float32, (128, 784))")	# Call self.decode(z) (line 4)
        self.assertEqual(str(id2type[44]), "class VAE")	# Name self (line 4)
        self.assertEqual(str(id2type[47]), "torch.Tensor(float32, (128, 20))")	# Name z (line 4)
        self.assertEqual(str(id2type[49]), "torch.Tensor(float32, (128, 20))")	# Name mu (line 4)
        self.assertEqual(str(id2type[51]), "torch.Tensor(float32, (128, 20))")	# Name logvar (line 4)
        # === function encode ===
        self.assertEqual(str(id2type[61]), "torch.Tensor(float32, (128, 400))")	# Name h1 (line 2)
        self.assertEqual(str(id2type[63]), "torch.Tensor(float32, (128, 400))")	# Call F.relu(self.fc1(x)) (line 2)
        self.assertEqual(str(id2type[68]), "torch.Tensor(float32, (128, 400))")	# Call self.fc1(x) (line 2)
        self.assertEqual(str(id2type[70]), "class VAE")	# Name self (line 2)
        self.assertEqual(str(id2type[73]), "torch.Tensor(float32, (128, 784))")	# Name x (line 2)
        self.assertEqual(str(id2type[76]), "(torch.Tensor(float32, (128, 20)), torch.Tensor(float32, (128, 20)))")	# Tuple (self.fc21(h1), self.fc22(h1)) (line 3)
        self.assertEqual(str(id2type[77]), "torch.Tensor(float32, (128, 20))")	# Call self.fc21(h1) (line 3)
        self.assertEqual(str(id2type[79]), "class VAE")	# Name self (line 3)
        self.assertEqual(str(id2type[82]), "torch.Tensor(float32, (128, 400))")	# Name h1 (line 3)
        self.assertEqual(str(id2type[84]), "torch.Tensor(float32, (128, 20))")	# Call self.fc22(h1) (line 3)
        self.assertEqual(str(id2type[86]), "class VAE")	# Name self (line 3)
        self.assertEqual(str(id2type[89]), "torch.Tensor(float32, (128, 400))")	# Name h1 (line 3)
        # === function reparameterize ===
        self.assertEqual(str(id2type[101]), "bool")	# Attribute self.training (line 2)
        self.assertEqual(str(id2type[102]), "class VAE")	# Name self (line 2)
        self.assertEqual(str(id2type[106]), "torch.Tensor(float32, (128, 20))")	# Name std (line 3)
        self.assertEqual(str(id2type[108]), "torch.Tensor(float32, (128, 20))")	# Call torch.exp(0.5 * logvar) (line 3)
        self.assertEqual(str(id2type[113]), "torch.Tensor(float32, (128, 20))")	# BinOp 0.5 * logvar (line 3)
        self.assertEqual(str(id2type[114]), "float")	# Constant 0.5 (line 3)
        self.assertEqual(str(id2type[116]), "torch.Tensor(float32, (128, 20))")	# Name logvar (line 3)
        self.assertEqual(str(id2type[119]), "torch.Tensor(float32, (128, 20))")	# Name eps (line 4)
        self.assertEqual(str(id2type[121]), "torch.Tensor(float32, (128, 20))")	# Call torch.randn_like(std) (line 4)
        self.assertEqual(str(id2type[126]), "torch.Tensor(float32, (128, 20))")	# Name std (line 4)
        self.assertEqual(str(id2type[129]), "torch.Tensor(float32, (128, 20))")	# Call eps.mul(std).add_(mu) (line 5)
        self.assertEqual(str(id2type[131]), "torch.Tensor(float32, (128, 20))")	# Call eps.mul(std) (line 5)
        self.assertEqual(str(id2type[133]), "torch.Tensor(float32, (128, 20))")	# Name eps (line 5)
        self.assertEqual(str(id2type[136]), "torch.Tensor(float32, (128, 20))")	# Name std (line 5)
        self.assertEqual(str(id2type[139]), "torch.Tensor(float32, (128, 20))")	# Name mu (line 5)
        self.assertEqual(str(id2type[142]), "torch.Tensor(float32, (128, 20))")	# Name mu (line 7)
        # === function decode ===
        self.assertEqual(str(id2type[151]), "torch.Tensor(float32, (128, 400))")	# Name h3 (line 2)
        self.assertEqual(str(id2type[153]), "torch.Tensor(float32, (128, 400))")	# Call F.relu(self.fc3(z)) (line 2)
        self.assertEqual(str(id2type[158]), "torch.Tensor(float32, (128, 400))")	# Call self.fc3(z) (line 2)
        self.assertEqual(str(id2type[160]), "class VAE")	# Name self (line 2)
        self.assertEqual(str(id2type[163]), "torch.Tensor(float32, (128, 20))")	# Name z (line 2)
        self.assertEqual(str(id2type[166]), "torch.Tensor(float32, (128, 784))")	# Call torch.sigmoid(self.fc4(h3)) (line 3)
        self.assertEqual(str(id2type[171]), "torch.Tensor(float32, (128, 784))")	# Call self.fc4(h3) (line 3)
        self.assertEqual(str(id2type[173]), "class VAE")	# Name self (line 3)
        self.assertEqual(str(id2type[176]), "torch.Tensor(float32, (128, 400))")	# Name h3 (line 3)
        # === END ASSERTIONS for VAE ===


def main():
    unittest.main()

if __name__ == '__main__':
    main()
