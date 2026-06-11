import unittest

import solas

class TensorTest(unittest.TestCase):
    def test_create_and_close_tensor(self) -> None:
        tensor = solas.Tensor()

        self.assertIsNotNone(tensor._handle)

        tensor.close()
        self.assertIsNone(tensor._handle)
    
    def test_default_tensor_is_scalar(self) -> None:
        tensor = solas.Tensor()

        self.assertEqual(tensor.rank, 0)
        self.assertEqual(tensor.shape, ())

        tensor.close()


if __name__ == "__main__":
    unittest.main()