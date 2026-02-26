/* CharFormer — TinyShakespeare Demo
 * Downloads TinyShakespeare, trains a character-level model, and generates text.
 * Demonstrates the Trails typed-tensor API with batch-agnostic CharFormer/CharRNN/CharGRU.
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <filesystem>
#include <cstdlib>
#include <cstdio>

#include <torch/torch.h>
#include "charformer.hpp"
#include "trails/trails.hpp"

using namespace trainium;
using namespace trails;

// ── Hyperparameters ──────────────────────────────────────────
constexpr int SeqLen    = 256;
constexpr int VocabSize = 256;
constexpr int ModelDim  = 128;
constexpr int NumHeads  = 4;
constexpr int FFDim     = 512;
constexpr int NLayers   = 4;

constexpr double LearningRate = 3e-4;
constexpr float  Temperature  = 0.8f;
constexpr int    GenLength    = 500;
constexpr int    LogEvery     = 10;

static const std::string kDataDir  = "./data/tinyshakespeare";
static const std::string kDataFile = kDataDir + "/input.txt";
static const std::string kURL =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

// ── Utilities ────────────────────────────────────────────────

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string format_number(size_t n) {
    std::string s = std::to_string(n);
    for (int i = (int)s.size() - 3; i > 0; i -= 3)
        s.insert(i, ",");
    return s;
}

// ── Download ─────────────────────────────────────────────────

void ensure_data() {
    if (std::filesystem::exists(kDataFile)) {
        auto sz = std::filesystem::file_size(kDataFile);
        std::cout << "📥 TinyShakespeare already cached (" << format_number(sz) << " bytes)\n";
        return;
    }
    std::cout << "📥 Downloading TinyShakespeare... " << std::flush;
    std::filesystem::create_directories(kDataDir);
    std::string cmd = "curl -sL -o " + kDataFile + " " + kURL;
    if (std::system(cmd.c_str()) != 0) {
        throw std::runtime_error("Download failed. Install curl or place input.txt manually.");
    }
    auto sz = std::filesystem::file_size(kDataFile);
    std::cout << "done (" << format_number(sz) << " bytes)\n";
}

// ── Batching ─────────────────────────────────────────────────

struct Batch {
    torch::Tensor x;  // (BatchSize, SeqLen)
    torch::Tensor y;  // (BatchSize, SeqLen)
};

Batch random_batch(const std::string& text, int batch_size, std::mt19937& rng) {
    std::uniform_int_distribution<size_t> dist(0, text.size() - SeqLen - 2);
    std::vector<int64_t> xs, ys;
    xs.reserve(batch_size * SeqLen);
    ys.reserve(batch_size * SeqLen);
    for (int b = 0; b < batch_size; b++) {
        size_t offset = dist(rng);
        for (int s = 0; s < SeqLen; s++) {
            xs.push_back(static_cast<uint8_t>(text[offset + s]));
            ys.push_back(static_cast<uint8_t>(text[offset + s + 1]));
        }
    }
    auto xten = torch::tensor(xs, torch::kLong).reshape({batch_size, SeqLen});
    auto yten = torch::tensor(ys, torch::kLong).reshape({batch_size, SeqLen});
    return {xten, yten};
}

// ── Training ─────────────────────────────────────────────────

template<typename Model>
void train(Model& model, const std::string& text, int batch_size, int num_epochs) {
    size_t steps_per_epoch = text.size() / (batch_size * SeqLen);
    std::mt19937 rng(42);
    auto device = model.parameters().front().device();

    torch::optim::AdamW optimizer(
        model.parameters(),
        torch::optim::AdamWOptions(LearningRate));

    std::cout << "\n🏋️  Training (SeqLen=" << SeqLen << ", ModelDim=" << ModelDim
              << ", " << NLayers << " layers)\n";

    for (int epoch = 1; epoch <= num_epochs; epoch++) {
        auto t0 = std::chrono::steady_clock::now();
        double epoch_loss = 0.0;
        size_t step = 0;

        std::printf("  Epoch %d/%d:\n", epoch, num_epochs);

        for (size_t s = 0; s < steps_per_epoch; s++) {
            auto batch = random_batch(text, batch_size, rng);
            batch.x = batch.x.to(device);
            batch.y = batch.y.to(device);
            optimizer.zero_grad();

            // Forward pass: BatchTensor<SeqLen> -> BatchTensor<SeqLen, VocabSize>
            auto input = BatchTensor<SeqLen>(batch.x);
            auto logits = model.forward(input);

            auto loss = language_model_loss(logits.t(), batch.y);

            loss.backward();
            optimizer.step();

            double lv = loss.template item<double>();
            epoch_loss += lv;
            step++;

            if (step % LogEvery == 0) {
                std::printf("    step %-4zu  loss: %.4f\n", step, lv);
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::printf("    ✓ Epoch %d complete — avg loss: %.4f, time: %.1fs\n\n",
                    epoch, epoch_loss / step, elapsed);
    }
}

// ── Text Generation ──────────────────────────────────────────

template<typename Model>
std::string generate(Model& model, const std::string& seed, int length, float temperature) {
    torch::NoGradGuard no_grad;
    std::string context = seed;

    for (int i = 0; i < length; i++) {
        // Take the last SeqLen bytes (or pad left with zeros if shorter)
        std::string input_str;
        if ((int)context.size() >= SeqLen) {
            input_str = context.substr(context.size() - SeqLen);
        } else {
            input_str = std::string(SeqLen - context.size(), '\0') + context;
        }

        // Convert to tensor: (1, SeqLen)
        std::vector<int64_t> bytes(SeqLen);
        for (int j = 0; j < SeqLen; j++) {
            bytes[j] = static_cast<uint8_t>(input_str[j]);
        }
        auto t = torch::tensor(bytes, torch::kLong).unsqueeze(0).to(model.parameters().front().device());
        auto input = BatchTensor<SeqLen>(t);

        // Forward pass
        auto logits = model.forward(input);  // BatchTensor<SeqLen, VocabSize>

        // Last position logits: (VocabSize,)
        auto last_logits = logits.t().index({0, -1});

        // Temperature-scaled sampling
        last_logits = last_logits / temperature;
        auto probs = torch::softmax(last_logits, 0);
        auto next_byte = torch::multinomial(probs, 1).template item<int64_t>();

        context += static_cast<char>(next_byte);
    }

    return context.substr(seed.size());
}

// ── Main ─────────────────────────────────────────────────────

void print_usage(const char* prog) {
    std::printf(
        "Usage: %s [OPTIONS]\n"
        "\n"
        "Train a character-level model on TinyShakespeare and generate text.\n"
        "\n"
        "Options:\n"
        "  --model MODEL   Model type: transformer, rnn, gru (default: transformer)\n"
        "  --batchsize N   Training batch size (default: 16)\n"
        "  --epochs N      Number of training epochs (default: 3)\n"
        "  --help          Show this help message and exit\n",
        prog);
}

int main(int argc, char* argv[]) {
    int batch_size = 16;
    int num_epochs = 3;
    std::string model_type = "transformer";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --model requires a value\n";
                return 1;
            }
            model_type = argv[++i];
            if (model_type != "transformer" && model_type != "rnn" && model_type != "gru") {
                std::cerr << "Error: --model must be one of: transformer, rnn, gru\n";
                return 1;
            }
        } else if (arg == "--batchsize") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --batchsize requires a value\n";
                return 1;
            }
            batch_size = std::atoi(argv[++i]);
            if (batch_size <= 0) {
                std::cerr << "Error: --batchsize must be a positive integer\n";
                return 1;
            }
        } else if (arg == "--epochs") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --epochs requires a value\n";
                return 1;
            }
            num_epochs = std::atoi(argv[++i]);
            if (num_epochs <= 0) {
                std::cerr << "Error: --epochs must be a positive integer\n";
                return 1;
            }
        } else {
            std::cerr << "Error: unknown option '" << arg << "'\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "\n"
              << "╔══════════════════════════════════════════╗\n"
              << "║   CharFormer — TinyShakespeare Demo      ║\n"
              << "╚══════════════════════════════════════════╝\n\n";

    // 1. Download data
    ensure_data();
    std::string text = read_file(kDataFile);
    std::cout << "📊 Dataset: " << format_number(text.size()) << " characters\n";

    // Common pipeline: create model, report params, train, generate
    auto run = [&](auto& model) {
        size_t n_params = 0;
        for (const auto& p : model.parameters()) n_params += p.numel();
        std::cout << "🧠 Model (" << model_type << "): " << format_number(n_params) << " parameters\n";
        std::cout << "⚙️  Config: batch_size=" << batch_size << ", epochs=" << num_epochs << "\n";

        train(model, text, batch_size, num_epochs);

        std::string seed = "ROMEO:\n";
        std::cout << "✍️  Generating text (temperature=" << Temperature
                  << ", seed=\"ROMEO:\")\n";
        std::cout << "────────────────────────────────────────────\n";
        std::cout << seed;
        std::string generated = generate(model, seed, GenLength, Temperature);
        std::cout << generated << "\n";
        std::cout << "────────────────────────────────────────────\n";
    };

    // 2. Create model and run
    if (model_type == "transformer") {
        auto model = CharFormer<SeqLen, VocabSize, ModelDim, NumHeads, FFDim, NLayers>();
        move_to_best_available_device(model);
        run(model);
    } else if (model_type == "rnn") {
        auto model = CharRNN<SeqLen, VocabSize, ModelDim, NLayers>();
        move_to_best_available_device(model);
        run(model);
    } else if (model_type == "gru") {
        auto model = CharGRU<SeqLen, VocabSize, ModelDim, NLayers>();
        move_to_best_available_device(model);
        run(model);
    }

    return 0;
}
