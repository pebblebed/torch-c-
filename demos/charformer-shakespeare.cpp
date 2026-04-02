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
#include <map>
#include <set>
#include <algorithm>

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

// Rough GPT-2-small-style proportions, scaled down for char-level TinyShakespeare.
constexpr int GPT2ModelDim = 768;
constexpr int GPT2NumHeads = 12;
constexpr int GPT2FFDim    = 3072;
constexpr int GPT2NLayers  = 12;

constexpr double LearningRate = 3e-4;
constexpr float  Temperature  = 0.8f;
constexpr int    GenLength    = 500;
constexpr int    LogEvery     = 10;

struct DatasetSpec {
    std::string hf_name;
    std::string hf_config;
    std::string split;
    std::string text_field;
    std::string local_dir;
    std::string local_file;
    std::string direct_url;
    bool via_huggingface = false;
};

const std::set<std::string> kDatasets = {
    "fineweb",
    "openwebtext",
    "shakespeare",
    "tinystories",
    "wikitext-103",
    "wikitext-2",
};

const std::map<std::string, DatasetSpec> kDatasetSpecs = {
    {"shakespeare", DatasetSpec{
        .hf_name = "",
        .hf_config = "",
        .split = "",
        .text_field = "",
        .local_dir = "./data/shakespeare",
        .local_file = "./data/shakespeare/input.txt",
        .direct_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        .via_huggingface = false,
    }},
    {"wikitext-2", DatasetSpec{
        .hf_name = "salesforce/wikitext",
        .hf_config = "wikitext-2-raw-v1",
        .split = "train",
        .text_field = "text",
        .local_dir = "./data/wikitext-2",
        .local_file = "./data/wikitext-2/train.txt",
        .direct_url = "",
        .via_huggingface = true,
    }},
    {"wikitext-103", DatasetSpec{
        .hf_name = "salesforce/wikitext",
        .hf_config = "wikitext-103-raw-v1",
        .split = "train",
        .text_field = "text",
        .local_dir = "./data/wikitext-103",
        .local_file = "./data/wikitext-103/train.txt",
        .direct_url = "",
        .via_huggingface = true,
    }},
    {"openwebtext", DatasetSpec{
        .hf_name = "Skylion007/openwebtext",
        .hf_config = "",
        .split = "train",
        .text_field = "text",
        .local_dir = "./data/openwebtext",
        .local_file = "./data/openwebtext/train.txt",
        .direct_url = "",
        .via_huggingface = true,
    }},
    {"tinystories", DatasetSpec{
        .hf_name = "roneneldan/TinyStories",
        .hf_config = "",
        .split = "train",
        .text_field = "text",
        .local_dir = "./data/tinystories",
        .local_file = "./data/tinystories/train.txt",
        .direct_url = "",
        .via_huggingface = true,
    }},
    {"fineweb", DatasetSpec{
        .hf_name = "HuggingFaceFW/fineweb-edu",
        .hf_config = "sample-10BT",
        .split = "train",
        .text_field = "text",
        .local_dir = "./data/fineweb",
        .local_file = "./data/fineweb/train.txt",
        .direct_url = "",
        .via_huggingface = true,
    }}
};

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

std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

std::string join_strings(const std::set<std::string>& values, const std::string& sep = ", ") {
    std::ostringstream ss;
    bool first = true;
    for (const auto& value : values) {
        if (!first) ss << sep;
        ss << value;
        first = false;
    }
    return ss.str();
}

const DatasetSpec& get_dataset_spec(const std::string& dataset_name) {
    auto it = kDatasetSpecs.find(dataset_name);
    if (it == kDatasetSpecs.end()) {
        throw std::runtime_error("Unknown dataset '" + dataset_name + "'");
    }
    return it->second;
}

void ensure_dataset_text_with_hf(const std::string& dataset_name, const DatasetSpec& spec) {
    std::filesystem::create_directories(spec.local_dir);
    std::cout << "📥 Downloading " << dataset_name << " from Hugging Face... " << std::flush;

    std::string py =
        "from datasets import load_dataset\n"
        "from pathlib import Path\n"
        "dataset_name = " + std::string("\"") + spec.hf_name + "\"\n" +
        "config_name = " + (spec.hf_config.empty() ? std::string("None") : (std::string("\"") + spec.hf_config + "\"")) + "\n" +
        "split_name = \"" + spec.split + "\"\n"
        "text_field = \"" + spec.text_field + "\"\n"
        "output_path = Path(\"" + spec.local_file + "\")\n"
        "output_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "ds = load_dataset(dataset_name, config_name, split=split_name) if config_name is not None else load_dataset(dataset_name, split=split_name)\n"
        "with output_path.open('w', encoding='utf-8') as f:\n"
        "    for row in ds:\n"
        "        text = row.get(text_field, '')\n"
        "        if text:\n"
        "            f.write(text)\n"
        "            if not text.endswith('\\n'):\n"
        "                f.write('\\n')\n";

    auto script_path = spec.local_dir + "/download_hf_dataset.py";
    {
        std::ofstream script(script_path, std::ios::binary);
        if (!script) throw std::runtime_error("Cannot write helper script " + script_path);
        script << py;
    }

    std::string cmd = "python3 " + shell_quote(script_path);
    if (std::system(cmd.c_str()) != 0) {
        throw std::runtime_error(
            "Hugging Face dataset download failed for '" + dataset_name +
            "'. Install Python packages: datasets and pyarrow.");
    }

    auto sz = std::filesystem::file_size(spec.local_file);
    std::cout << "done (" << format_number(sz) << " bytes)\n";
}

void ensure_dataset_data(const std::string& dataset_name) {
    const auto& spec = get_dataset_spec(dataset_name);
    if (std::filesystem::exists(spec.local_file)) {
        auto sz = std::filesystem::file_size(spec.local_file);
        std::cout << "📥 " << dataset_name << " already cached (" << format_number(sz) << " bytes)\n";
        return;
    }

    if (!spec.direct_url.empty()) {
        std::cout << "📥 Downloading " << dataset_name << "... " << std::flush;
        std::filesystem::create_directories(spec.local_dir);
        std::string cmd = "curl -sL -o " + shell_quote(spec.local_file) + " " + shell_quote(spec.direct_url);
        if (std::system(cmd.c_str()) != 0) {
            throw std::runtime_error("Download failed for dataset '" + dataset_name + "'.");
        }
        auto sz = std::filesystem::file_size(spec.local_file);
        std::cout << "done (" << format_number(sz) << " bytes)\n";
        return;
    }

    if (spec.via_huggingface) {
        ensure_dataset_text_with_hf(dataset_name, spec);
        return;
    }

    throw std::runtime_error("No download strategy configured for dataset '" + dataset_name + "'.");
}

// ── Batching ─────────────────────────────────────────────────

struct Batch {
    BatchTensor<SeqLen> x;
    BatchTensor<SeqLen> y;
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
    return {BatchTensor<SeqLen>(xten), BatchTensor<SeqLen>(yten)};
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
            batch.x = BatchTensor<SeqLen>(batch.x.t().to(device));
            batch.y = BatchTensor<SeqLen>(batch.y.t().to(device));
            optimizer.zero_grad();

            auto logits = model.forward(batch.x);
            auto loss = language_model_loss(logits, batch.y);

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
        "  --model MODEL   Model type: transformer, rnn, gru, gpt2 (default: transformer)\n"
        "  --dataset NAME  Dataset: fineweb, openwebtext, shakespeare, tinystories, wikitext-103, wikitext-2 (default: shakespeare)\n"
        "  --batchsize N   Training batch size (default: 16)\n"
        "  --epochs N      Number of training epochs (default: 3)\n"
        "  --help          Show this help message and exit\n",
        prog);
}

int main(int argc, char* argv[]) {
    int batch_size = 16;
    int num_epochs = 3;
    std::string model_type = "transformer";
    std::string dataset_name = "shakespeare";
    const std::set<std::string> models = {"gpt2", "gru", "rnn", "transformer"};

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
            if (!models.contains(model_type)) {
                std::cerr << "Error: --model must be one of: " << join_strings(models) << "\n";
                return 1;
            }
        } else if (arg == "--dataset") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --dataset requires a value\n";
                return 1;
            }
            dataset_name = argv[++i];
            if (!kDatasets.contains(dataset_name)) {
                std::cerr << "Error: --dataset must be one of: " << join_strings(kDatasets) << "\n";
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
    ensure_dataset_data(dataset_name);
    const auto& dataset_spec = get_dataset_spec(dataset_name);
    std::string text = read_file(dataset_spec.local_file);
    std::cout << "📊 Dataset (" << dataset_name << "): " << format_number(text.size()) << " characters\n";

    // Common pipeline: create model, report params, train, generate
    auto run = [&](auto& model) {
        size_t n_params = 0;
        for (const auto& p : model.parameters()) n_params += p.numel();
        std::cout << "🧠 Model (" << model_type << "): " << format_number(n_params) << " parameters\n";
        std::cout << "⚙️  Config: dataset=" << dataset_name << ", batch_size=" << batch_size << ", epochs=" << num_epochs << "\n";

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
    } else if (model_type == "gpt2") {
        auto model = CharGPT2<SeqLen, VocabSize, GPT2ModelDim, GPT2NumHeads, GPT2FFDim, GPT2NLayers>();
        move_to_best_available_device(model);
        run(model);
    }

    return 0;
}
