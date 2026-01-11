#include <gtest/gtest.h>
#include "perfcli/core/Types.hpp"
#include <sstream>

TEST(DataType, SizeAndStringConversion) {
  EXPECT_EQ(perfcli::data_type_size(perfcli::DataType::Float32), sizeof(float));
  EXPECT_EQ(perfcli::data_type_size(perfcli::DataType::Float16), 2);
  EXPECT_EQ(perfcli::data_type_size(perfcli::DataType::BFloat16), 2);
  EXPECT_EQ(perfcli::data_type_size(perfcli::DataType::Int8), 1);
  EXPECT_EQ(perfcli::data_type_size(perfcli::DataType::Int32), sizeof(int32_t));

  EXPECT_EQ(perfcli::data_type_to_string(perfcli::DataType::Float32), "fp32");
  EXPECT_EQ(perfcli::data_type_to_string(perfcli::DataType::Float16), "fp16");
  EXPECT_EQ(perfcli::data_type_to_string(perfcli::DataType::BFloat16), "bf16");
  EXPECT_EQ(perfcli::data_type_to_string(perfcli::DataType::Int8), "int8");
  EXPECT_EQ(perfcli::data_type_to_string(perfcli::DataType::Int32), "int32");

  EXPECT_EQ(perfcli::string_to_data_type("fp32"), perfcli::DataType::Float32);
  EXPECT_EQ(perfcli::string_to_data_type("fp16"), perfcli::DataType::Float16);
  EXPECT_EQ(perfcli::string_to_data_type("bf16"), perfcli::DataType::BFloat16);
  EXPECT_EQ(perfcli::string_to_data_type("int8"), perfcli::DataType::Int8);
  EXPECT_EQ(perfcli::string_to_data_type("int32"), perfcli::DataType::Int32);
}

TEST(Direction, StringConversion) {
  EXPECT_EQ(perfcli::direction_to_string(perfcli::Direction::HostToDevice), "H2D");
  EXPECT_EQ(perfcli::direction_to_string(perfcli::Direction::DeviceToHost), "D2H");
  EXPECT_EQ(perfcli::direction_to_string(perfcli::Direction::DeviceToDevice), "D2D");
}

TEST(BenchmarkResult, DefaultConstruction) {
  perfcli::BenchmarkResult result;

  EXPECT_TRUE(result.benchmark_name.empty());
  EXPECT_TRUE(result.params.empty());
  EXPECT_TRUE(result.raw_samples_us.empty());
  EXPECT_FALSE(result.success);
}

TEST(BenchmarkResult, JsonSerialization) {
  perfcli::BenchmarkResult result;
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
  perfcli::BenchmarkSpec spec;

  EXPECT_TRUE(spec.name.empty());
  EXPECT_TRUE(spec.description.empty());
  EXPECT_TRUE(spec.parameters.empty());
  EXPECT_TRUE(spec.default_params.empty());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
