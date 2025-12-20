import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import unittest
import shutil
# 直接导入，因为 visualizer 已经处理了路径，这里不需要 hack path
from models.model import CDSPConfig, CDSPModel
from analysis.visualizer import SystemMonitor


class TestVisualizationPipeline(unittest.TestCase):
    def setUp(self):
        # 动态获取 visualizer 应该生成的真实绝对路径
        # 预期路径: .../CDSP-MoE/analysis/pic/test_pipeline
        # 我们通过实例化 Monitor 来获取这个路径，而不是硬编码
        self.monitor = SystemMonitor(exp_name="test_pipeline")
        self.real_save_dir = self.monitor.save_dir

        # 如果存在旧数据先清理，保证测试纯净
        if os.path.exists(self.real_save_dir):
            shutil.rmtree(self.real_save_dir)
            os.makedirs(self.real_save_dir, exist_ok=True)

    def test_monitor_integration(self):
        print("\n=== Testing Visualization Pipeline (Absolute Path) ===")
        print(f"Target Directory: {self.real_save_dir}")

        # 1. 建立极简模型
        config = CDSPConfig(
            vocab_size=10, d_model=8, n_layers=1, d_base=16,
            num_experts=2, num_tasks=2, moe_top_k=1
        )
        model = CDSPModel(config)

        # 2. 模拟几步数据
        alpha = model.layers[0].moe.topology.alpha
        for step in range(3):
            self.monitor.log_step(step, {'loss': 1.0})
            self.monitor.capture_snapshot(step, alpha_matrix=alpha)

        # 3. 生成图片
        filename = "test_topo_abs.png"
        self.monitor.plot_topology_evolution(filename=filename)

        # 4. 验证：文件必须出现在 analysis/pic/test_pipeline 下
        expected_file = os.path.join(self.real_save_dir, filename)

        # 断言文件存在
        self.assertTrue(os.path.exists(expected_file),
                        f"Image not found at expected path: {expected_file}")

        # 断言文件大小 > 0 (不是空文件)
        self.assertGreater(os.path.getsize(expected_file), 0, "Generated image is empty!")

        print(f"SUCCESS: Image generated at {expected_file}")

    def tearDown(self):
        # 测试通过后清理垃圾文件 (如果想保留查看结果，可以注释掉这行)
        if os.path.exists(self.real_save_dir):
            shutil.rmtree(self.real_save_dir)
            pass


if __name__ == '__main__':
    unittest.main()