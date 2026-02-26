#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <cstring>
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

struct ScopedTempDir {
    std::filesystem::path path;

    ScopedTempDir() {
        auto base = std::filesystem::temp_directory_path();
        path = base / std::filesystem::path("trails_dataset_dir_test_" + std::to_string(::getpid()));
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }

    ~ScopedTempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
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

TEST(MMappedFileTests, MissingPathPreservesOpenErrorMessage) {
    auto missing_path = std::filesystem::temp_directory_path()
        / std::filesystem::path("trails_missing_" + std::to_string(::getpid()) + "_nope.bin");
    std::filesystem::remove(missing_path);

    try {
        trainium::MMappedFile f(missing_path.string());
        FAIL() << "Expected constructor to throw for missing file";
    } catch (const std::runtime_error& e) {
        auto message = std::string(e.what());
        EXPECT_NE(message.find(std::strerror(ENOENT)), std::string::npos);
    }
}

TEST(DatasetDirTests, NestedTraversalAccumulatesAllSubdirectories) {
    ScopedTempDir temp_dir;
    auto sub_a = temp_dir.path / "a";
    auto sub_b = temp_dir.path / "b";
    std::filesystem::create_directories(sub_a);
    std::filesystem::create_directories(sub_b);

    {
        std::ofstream out(sub_a / "data_a.bin", std::ios::binary | std::ios::trunc);
        const std::string bytes = "ABCDE"; // len=5 => windows(len,n_ctx=3)=3
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    {
        std::ofstream out(sub_b / "data_b.bin", std::ios::binary | std::ios::trunc);
        const std::string bytes = "ABCDEFG"; // len=7 => windows(len,n_ctx=3)=5
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    trainium::DatasetDir dataset(temp_dir.path.string(), /*batch_size=*/1, /*n_ctx=*/3);
    auto sz = dataset.size();
    ASSERT_TRUE(sz.has_value());
    EXPECT_EQ(sz.value(), 8u);
}
