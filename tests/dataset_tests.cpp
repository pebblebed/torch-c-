#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../dataset_dir.hpp"

namespace {

struct ScopedTempFile {
    std::filesystem::path path;

    explicit ScopedTempFile(const std::vector<uint8_t>& bytes) {
        auto dir = std::filesystem::temp_directory_path();
        path = dir / std::filesystem::path("trails_dataset_test_" + std::to_string(::getpid()) + ".bin");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    ~ScopedTempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

} // namespace

TEST(DatasetFileTests, ExactContextLengthProvidesSingleExample) {
    constexpr size_t n_ctx = 5;
    ScopedTempFile file({10, 11, 12, 13, 14}); // size == n_ctx

    trainium::DatasetFile dataset(file.path.string(), /*batch_size=*/1, n_ctx);
    auto sz = dataset.size();
    ASSERT_TRUE(sz.has_value());
    ASSERT_EQ(sz.value(), 1u);

    auto ex = dataset.get(0);
    ASSERT_EQ(ex.data.dim(), 1);
    ASSERT_EQ(ex.target.dim(), 1);
    ASSERT_EQ(ex.data.size(0), static_cast<int64_t>(n_ctx - 1));
    ASSERT_EQ(ex.target.size(0), static_cast<int64_t>(n_ctx - 1));
    EXPECT_EQ(ex.data[0].item<int64_t>(), 10);
    EXPECT_EQ(ex.target[0].item<int64_t>(), 11);
}
