/* CharFormer — TinyShakespeare Demo
 * Downloads TinyShakespeare, trains a character-level transformer, and generates text.
 * Demonstrates the Trails typed-tensor API with batch-agnostic CharFormer.
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
#include "trails.hpp"

using namespace trainium;
using namespace trails;

// ── Hyperparameters ──────────────────────────────────────────
constexpr int SeqLen    = 256;
constexpr int VocabSize = 256;
constexpr int ModelDim  = 128;
constexpr int NumHeads  = 4;
constexpr int FFDim     = 512;
constexpr int NLayers   = 4;

constexpr int    BatchSize  = 16;
constexpr int    NumEpochs  = 3;
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

Batch random_batch(const std::string& text, std::mt19937& rng) {
    std::uniform_int_distribution<size_t> dist(0, text.size() - SeqLen - 2);
    std::vector<int64_t> xs, ys;
    xs.reserve(BatchSize * SeqLen);
    ys.reserve(BatchSize * SeqLen);
    for (int b = 0; b < BatchSize; b++) {
        size_t offset = dist(rng);
        for (int s = 0; s < SeqLen; s++) {
            xs.push_back(static_cast<uint8_t>(text[offset + s]));
            ys.push_back(static_cast<uint8_t>(text[offset + s + 1]));
        }
    }
    auto xten = torch::tensor(xs, torch::kLong).reshape({BatchSize, SeqLen});
    auto yten = torch::tensor(ys, torch::kLong).reshape({BatchSize, SeqLen});
    return {xten, yten};
}

// ── Training ─────────────────────────────────────────────────

void train(CharFormer<SeqLen, VocabSize, ModelDim, NumHeads, FFDim, NLayers>& model,
           const std::string& text) {
    size_t steps_per_epoch = text.size() / (BatchSize * SeqLen);
    std::mt19937 rng(42);

    torch::optim::AdamW optimizer(
        model.parameters(),
        torch::optim::AdamWOptions(LearningRate));

    std::cout << "\n🏋️  Training (SeqLen=" << SeqLen << ", ModelDim=" << ModelDim
              << ", " << NumHeads << " heads, " << NLayers << " layers)\n";

    for (int epoch = 1; epoch <= NumEpochs; epoch++) {
        auto t0 = std::chrono::steady_clock::now();
        double epoch_loss = 0.0;
        size_t step = 0;

        std::printf("  Epoch %d/%d:\n", epoch, NumEpochs);

        for (size_t s = 0; s < steps_per_epoch; s++) {
            auto batch = random_batch(text, rng);
            optimizer.zero_grad();

            // Forward pass: BatchTensor<SeqLen> -> BatchTensor<SeqLen, VocabSize>
            auto input = Tensor<BatchSize, SeqLen>(batch.x).unbatch();
            auto logits = model.forward(input);

            // Flatten for cross_entropy: (B*SeqLen, VocabSize) vs (B*SeqLen,)
            auto logits_flat = logits.t().reshape({BatchSize * SeqLen, VocabSize});
            auto target_flat = batch.y.reshape({BatchSize * SeqLen});
            auto loss = torch::nn::functional::cross_entropy(logits_flat, target_flat);

            loss.backward();
            optimizer.step();

            double lv = loss.item<double>();
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

std::string generate(CharFormer<SeqLen, VocabSize, ModelDim, NumHeads, FFDim, NLayers>& model,
                     const std::string& seed, int length, float temperature) {
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
        auto t = torch::tensor(bytes, torch::kLong).unsqueeze(0);
        auto input = BatchTensor<SeqLen>(t);

        // Forward pass
        auto logits = model.forward(input);  // BatchTensor<SeqLen, VocabSize>

        // Last position logits: (VocabSize,)
        auto last_logits = logits.t().index({0, -1});

        // Temperature-scaled sampling
        last_logits = last_logits / temperature;
        auto probs = torch::softmax(last_logits, 0);
        auto next_byte = torch::multinomial(probs, 1).item<int64_t>();

        context += static_cast<char>(next_byte);
    }

    return context.substr(seed.size());
}

// ── Main ─────────────────────────────────────────────────────

int main() {
    std::cout << "\n"
              << "╔══════════════════════════════════════════╗\n"
              << "║   CharFormer — TinyShakespeare Demo      ║\n"
              << "╚══════════════════════════════════════════╝\n\n";

    // 1. Download data
    ensure_data();
    std::string text = read_file(kDataFile);
    std::cout << "📊 Dataset: " << format_number(text.size()) << " characters\n";

    // 2. Create model
    auto model = CharFormer<SeqLen, VocabSize, ModelDim, NumHeads, FFDim, NLayers>();
    size_t n_params = 0;
    for (const auto& p : model.parameters()) n_params += p.numel();
    std::cout << "🧠 Model: " << format_number(n_params) << " parameters\n";

    // 3. Train
    train(model, text);

    // 4. Generate
    std::string seed = "ROMEO:\n";
    std::cout << "✍️  Generating text (temperature=" << Temperature
              << ", seed=\"ROMEO:\")\n";
    std::cout << "────────────────────────────────────────────\n";
    std::cout << seed;
    std::string generated = generate(model, seed, GenLength, Temperature);
    std::cout << generated << "\n";
    std::cout << "────────────────────────────────────────────\n";

    return 0;
}
