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

    def test_create_tensor_with_shape(self) -> None:
        tensor = solas.Tensor((2, 3))

        self.assertEqual(tensor.rank, 2)
        self.assertEqual(tensor.shape, (2, 3))

        tensor.close()
    
    def test_rejects_negative_shape_dimension(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "negative shape dimension"):
            solas.Tensor((-1,))
    
    def test_successful_tensor_create_clears_last_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "negative shape dimension"):
            solas.Tensor((-1,))

        tensor = solas.Tensor((2, 3))

        self.assertEqual(solas.last_error(), "")

        tensor.close()


if __name__ == "__main__":
    unittest.main()