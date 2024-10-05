#pragma once

#include <torch/torch.h>
#include <tuple>
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
    const size_t size;
    const uint8_t* data;

    MMappedFile(const std::string& path)
    : fd(::open(path.c_str(), O_RDONLY))
    , size(lseek(fd, 0, SEEK_END))
    , data(static_cast<uint8_t*>(::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0)))
    {
        if (fd == -1) {
            throw std::runtime_error("Failed to open file: " + std::string(strerror(errno)));
        }
        ::close(fd);
        if (size != 0 && data == MAP_FAILED) {
            if (size == 0) {
                return;
            }
            throw std::runtime_error("Failed to mmap file: " + path + " " + std::string(strerror(errno)));
        }
    }

    MMappedFile(MMappedFile&& other)
    : fd(other.fd), size(other.size), data(other.data) {
        other.data = nullptr;
    }

    MMappedFile& operator=(MMappedFile&& other) {
        if (this != &other) {
            this->~MMappedFile();
            new (this) MMappedFile(std::move(other));
        }
        return *this;
    }

    ~MMappedFile() {
        if (data != nullptr && data != MAP_FAILED) {
            ::munmap(const_cast<uint8_t*>(data), size);
        }
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
    protected:
    void traverse(const std::string& path);
public:
    DatasetDir(const std::string& path, size_t batch_size, size_t n_ctx);

    Example get(size_t index) override;
    torch::optional<size_t> size() const override;

private:
    std::vector<DatasetFile> files;
    size_t total_size;
    size_t batch_size;
    size_t n_ctx;
};

class SubsetDataset: public torch::data::Dataset<SubsetDataset> {
public:
    SubsetDataset(DatasetDir& parent, int start_idx, int end_idx)
    : parent(parent)
    , start(start_idx)
    , finish(end_idx) { }

    Example get(size_t index) override;
    torch::optional<size_t> size() const override {
        return finish - start;
    }
private:
    DatasetDir& parent;
    int start;
    int finish;
};

static inline
std::tuple<SubsetDataset, SubsetDataset, SubsetDataset>
train_test_val_split(DatasetDir& parent, float test_pct=0.1, float val_pct=0.1) {
    auto train_pct = 1.0 - (test_pct + val_pct);
    size_t train_end = parent.size().value_or(0) * train_pct;
    auto train = SubsetDataset(parent, 0, train_end);

    // No overlap here; train_end is not part of the train set.
    size_t val_start = train_end;
    size_t val_end = val_start + parent.size().value_or(0) * val_pct;
    auto val = SubsetDataset(parent, val_start, val_end);

    size_t test_start = val_end;
    size_t test_end = test_start + parent.size().value_or(0) * test_pct;
    auto test = SubsetDataset(parent, test_start, test_end);
    return std::make_tuple(train, test, val);
}

}