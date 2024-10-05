#pragma once

#include <torch/torch.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>

namespace trainium {

typedef torch::Tensor X;
typedef torch::Tensor Y;
typedef torch::data::Example<X, Y> Example;

struct MMappedFile {
    const int fd;
    const uint8_t* data;
    const size_t size;

    MMappedFile(const std::string& path)
    : fd(::open(path.c_str(), O_RDONLY))
    , data(static_cast<uint8_t*>(::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0)))
    , size(lseek(fd, 0, SEEK_END))
    {
        if (fd == -1) {
            throw std::runtime_error("Failed to open file: " + std::string(strerror(errno)));
        }
        if (data == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("Failed to mmap file: " + std::string(strerror(errno)));
        }
    }

    ~MMappedFile() {
        ::munmap(const_cast<uint8_t*>(data), size);
        ::close(fd);
    }
};


class DatasetFile : public torch::data::Dataset<DatasetFile> {
public:
    DatasetFile(const std::string& path, size_t batch_size, size_t n_ctx) 
    : file(path), batch_size(batch_size), n_ctx(n_ctx) {
    }

    Example get(size_t index) override;
    torch::optional<size_t> size() const override;

private:
    MMappedFile file;
    size_t batch_size;
    size_t n_ctx;
};

class DatasetDir : public torch::data::Dataset<DatasetDir> {
public:
    DatasetDir(const std::string& path, size_t batch_size, size_t n_ctx);

    Example get(size_t index) override;
    torch::optional<size_t> size() const override;

private:
    std::vector<DatasetFile> files;
    size_t batch_size;
    size_t n_ctx;
};

}
