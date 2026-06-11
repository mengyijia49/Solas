
#使用 Python 标准库测试框架，不引入第三方依赖。
import unittest

#导入你刚才写的 Python 前端包。
import solas


class VersionTest(unittest.TestCase):
    def test_version(self) -> None:
        self.assertEqual(solas.version(), "0.0.0")


if __name__ == "__main__":
    unittest.main()