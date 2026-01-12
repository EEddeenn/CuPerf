#include <gtest/gtest.h>
#include "cuperf/core/Types.hpp"
#include <sstream>

TEST(DataType, SizeAndStringConversion) {
  EXPECT_EQ(cuperf::data_type_size(cuperf::DataType::Float32), sizeof(float));
  EXPECT_EQ(cuperf::data_type_size(cuperf::DataType::Float16), 2);
  EXPECT_EQ(cuperf::data_type_size(cuperf::DataType::BFloat16), 2);
  EXPECT_EQ(cuperf::data_type_size(cuperf::DataType::Int8), 1);
  EXPECT_EQ(cuperf::data_type_size(cuperf::DataType::Int32), sizeof(int32_t));

  EXPECT_EQ(cuperf::data_type_to_string(cuperf::DataType::Float32), "fp32");
  EXPECT_EQ(cuperf::data_type_to_string(cuperf::DataType::Float16), "fp16");
  EXPECT_EQ(cuperf::data_type_to_string(cuperf::DataType::BFloat16), "bf16");
  EXPECT_EQ(cuperf::data_type_to_string(cuperf::DataType::Int8), "int8");
  EXPECT_EQ(cuperf::data_type_to_string(cuperf::DataType::Int32), "int32");

  EXPECT_EQ(cuperf::string_to_data_type("fp32"), cuperf::DataType::Float32);
  EXPECT_EQ(cuperf::string_to_data_type("fp16"), cuperf::DataType::Float16);
  EXPECT_EQ(cuperf::string_to_data_type("bf16"), cuperf::DataType::BFloat16);
  EXPECT_EQ(cuperf::string_to_data_type("int8"), cuperf::DataType::Int8);
  EXPECT_EQ(cuperf::string_to_data_type("int32"), cuperf::DataType::Int32);
}

TEST(Direction, StringConversion) {
  EXPECT_EQ(cuperf::direction_to_string(cuperf::Direction::HostToDevice), "H2D");
  EXPECT_EQ(cuperf::direction_to_string(cuperf::Direction::DeviceToHost), "D2H");
  EXPECT_EQ(cuperf::direction_to_string(cuperf::Direction::DeviceToDevice), "D2D");
}

TEST(BenchmarkResult, DefaultConstruction) {
  cuperf::BenchmarkResult result;

  EXPECT_TRUE(result.benchmark_name.empty());
  EXPECT_TRUE(result.params.empty());
  EXPECT_TRUE(result.raw_samples_us.empty());
  EXPECT_FALSE(result.success);
}

TEST(BenchmarkResult, JsonSerialization) {
  cuperf::BenchmarkResult result;
  result.benchmark_name = "test_benchmark";
  result.device_index = 0;
  result.success = true;
  result.median_us = 42.0;
  result.p95_us = 50.0;
  result.p99_us = 55.0;
  result.mean_us = 43.5;
  result.stddev_us = 5.0;
  result.metrics["throughput"] = 100.0;

  std::string json = result.to_json(false);

  EXPECT_FALSE(json.empty());
  EXPECT_NE(json.find("test_benchmark"), std::string::npos);
  EXPECT_NE(json.find("42.00"), std::string::npos);
}

TEST(BenchmarkSpec, DefaultConstruction) {
  cuperf::BenchmarkSpec spec;

  EXPECT_TRUE(spec.name.empty());
  EXPECT_TRUE(spec.description.empty());
  EXPECT_TRUE(spec.parameters.empty());
  EXPECT_TRUE(spec.default_params.empty());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
