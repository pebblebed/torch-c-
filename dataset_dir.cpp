#include "dataset_dir.hpp"
#include <filesystem>
#include <vector>
#include <cstdint>
#include <unistd.h>
#include <fcntl.h>
#include <cassert>
#include <sys/mman.h>

namespace trainium {
// Some file-handling utilities

void null_deleter(void* data) {
    // Do nothing
}

const torch::Tensor make_tensor(const uint8_t* data, size_t size) {
    std::vector<int64_t> shape = {static_cast<int64_t>(size)};
    std::vector<int64_t> strides = {sizeof(uint8_t)};
    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    return torch::from_blob(
        const_cast<uint8_t *>(data),
        at::IntArrayRef(shape),
        at::IntArrayRef(strides),
        [](void*){},
        options).to(torch::kInt32);
}

static inline uint64_t dumb_hash(uint64_t input) {
    return (input * 2654435761) >> 32;
}

Example DatasetFile::get(size_t index) {
    auto offset = dumb_hash(index) % (file.size - n_ctx);
    auto data = file.data + offset;
    torch::Tensor x = make_tensor(data, n_ctx - 1);
    torch::Tensor y = make_tensor(data + n_ctx, 1);
    return Example(x, y);
}

torch::optional<size_t> DatasetFile::size() const {
    if (file.size < n_ctx) {
        return torch::nullopt;
    }
    return (file.size - n_ctx) / batch_size;
}

DatasetDir::DatasetDir(const std::string& path, size_t batch_size, size_t n_ctx)
: batch_size(batch_size), n_ctx(n_ctx) {
    traverse(path);
    std::cerr << "Found " << files.size() << " files" << std::endl;
}

void DatasetDir::traverse(const std::string& path) {
    total_size = 0;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            auto file = DatasetFile { entry.path().string(), batch_size, n_ctx };
            if (!file.size().has_value()) {
                continue;
            }
            files.push_back(std::move(file));
            total_size += files.back().size().value();
            std::cerr << "File " << entry.path().string() << " size: " << files.back().size().value() << std::endl;
            std::cerr << "Files: " << files.size() << " Total size: " << total_size << std::endl;
            continue;
        }
        if (entry.is_directory()) {
            traverse(entry.path().string());
        }
    }
}

Example DatasetDir::get(size_t index) {
    auto &file = files[dumb_hash(index) % files.size()];
    return file.get(index);
}

torch::optional<size_t> DatasetDir::size() const {
    return total_size;
}


Example SubsetDataset::get(size_t index) {
    index = (index + start) % (finish - start);
    return parent.get(start + index);
}

}
