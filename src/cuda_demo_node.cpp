#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <vector>

extern "C" void launch_vector_add(const float* A, const float* B, float* C,
                                  int N);

class CudaDemoNode : public rclcpp::Node {
 public:
  CudaDemoNode() : Node("cuda_demo_node") {
    RCLCPP_INFO(this->get_logger(), "CUDA Demo Node Started");

    const int N = 1024;
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N);

    // CUDAカーネルの呼び出し
    launch_vector_add(A.data(), B.data(), C.data(), N);

    // 結果の表示
    for (int i = 0; i < 10; ++i) {
      RCLCPP_INFO(this->get_logger(), "C[%d] = %f", i, C[i]);
    }
  }
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CudaDemoNode>());
  rclcpp::shutdown();
  return 0;
}
