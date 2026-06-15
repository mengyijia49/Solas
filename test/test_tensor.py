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

    #普通 shape 的测试：
    def test_tensor_numel(self) -> None:
        scalar = solas.Tensor()
        matrix = solas.Tensor((2, 3))

        self.assertEqual(scalar.numel, 1)
        self.assertEqual(matrix.numel, 6)

        scalar.close()
        matrix.close()
    
    def test_tensor_data_size_matches_numel(self) -> None:
        scalar = solas.Tensor()
        matrix = solas.Tensor((2, 3))
        empty = solas.Tensor((0, 3))

        self.assertEqual(scalar.data_size, scalar.numel)
        self.assertEqual(matrix.data_size, matrix.numel)
        self.assertEqual(empty.data_size, empty.numel)

        scalar.close()
        matrix.close()
        empty.close()
    
    #bnytes测试：
    def test_tensor_nbytes(self) -> None:
        scalar = solas.Tensor()
        matrix = solas.Tensor((2, 3))
        empty = solas.Tensor((0, 3))

        self.assertEqual(scalar.nbytes, 4)
        self.assertEqual(matrix.nbytes, 24)
        self.assertEqual(empty.nbytes, 0)

        scalar.close()
        matrix.close()
        empty.close()

    #dtype测试-默认 dtype 是 float32：
    def test_tensor_dtype_defaults_to_float32(self) -> None:
        tensor = solas.Tensor((2, 3))

        self.assertEqual(tensor.dtype, "float32")

        tensor.close()

    #非法 dtype 的测试-Python 层会拒绝暂不支持的 dtype：
    def test_rejects_unsupported_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported dtype"):
            solas.Tensor((2, 3), dtype="int8")

    #验证数据变成了 1.5，可能对某些小数有浮点误差，所以使用assertAlmostEqual。
    def test_tensor_fill_float32(self) -> None:
        tensor = solas.Tensor((2, 3))

        tensor.fill(1.5)

        self.assertEqual(tensor.data_size, 6)
        self.assertAlmostEqual(tensor.get(0), 1.5)
        self.assertAlmostEqual(tensor.get(5), 1.5)


        tensor.close()

    #set测试：
    def test_tensor_set_float32(self) -> None:
        tensor = solas.Tensor((2, 3))

        tensor.fill(0.0)
        tensor.set(4, 9.0)

        self.assertAlmostEqual(tensor.get(0), 0.0)
        self.assertAlmostEqual(tensor.get(4), 9.0)

        tensor.close()

    
    def test_scalar_tensor_to_list(self) -> None:
        tensor = solas.Tensor()
        tensor.fill(3.5)

        self.assertAlmostEqual(tensor.to_list(), 3.5)

        tensor.close()

    # 一维 Tensor 测试：
    def test_1d_tensor_to_list(self) -> None:
        tensor = solas.Tensor((3,))
        tensor.fill(1.0)
        tensor.set(1, 2.0)

        self.assertEqual(tensor.to_list(), [1.0, 2.0, 1.0])

        tensor.close()

    # 二维 Tensor 测试：
    def test_2d_tensor_to_list(self) -> None:
        tensor = solas.Tensor((2, 3))
        tensor.fill(1.0)
        tensor.set(4, 9.0)

        self.assertEqual(tensor.to_list(), [[1.0, 1.0, 1.0], [1.0, 9.0, 1.0]])

        tensor.close()
    
    #set越界写入测试：
    def test_tensor_set_rejects_out_of_range_index(self) -> None:
        tensor = solas.Tensor((2, 3))

        with self.assertRaisesRegex(RuntimeError, "out-of-range index"):
            tensor.set(6, 1.0)

        tensor.close()
    
    #越界读取测试：
    def test_tensor_get_rejects_out_of_range_index(self) -> None:
        tensor = solas.Tensor((2, 3))

        with self.assertRaisesRegex(RuntimeError, "out-of-range index"):
            tensor.get(6)

        tensor.close()

    #包含 0 维度的测试：
    def test_tensor_numel_with_zero_dimension(self) -> None:
        tensor = solas.Tensor((0, 3))

        self.assertEqual(tensor.rank, 2)
        self.assertEqual(tensor.shape, (0, 3))
        self.assertEqual(tensor.numel, 0)

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