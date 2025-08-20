#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

using std::size_t;

static inline size_t idx(size_t i, size_t j, size_t ld) { return i * ld + j; }

struct Img {
    size_t h{}, w{};
    std::vector<double> d;
    Img() = default;
    Img(size_t H, size_t W) : h(H), w(W), d(H * W) {}
    inline double& operator()(size_t i, size_t j) { return d[idx(i,j,w)]; }
    inline const double& operator()(size_t i, size_t j) const { return d[idx(i,j,w)]; }
};

static void fill_deterministic(Img& M) {
    for (size_t i = 0; i < M.h; ++i)
        for (size_t j = 0; j < M.w; ++j)
            M(i,j) = ((i * 1315423911ull + j * 2654435761ull) & 0xFFFFF) / double(0xFFFFF);
}

static void worker(const Img& in, const Img& ker, Img& out, size_t row_begin, size_t row_end) {
    const size_t H = in.h, W = in.w, K = ker.h;
    for (size_t i = row_begin; i < row_end; ++i) {
        for (size_t j = 0; j < out.w; ++j) {
            double acc = 0.0;
            for (size_t ki = 0; ki < K; ++ki) {
                const double* inrow = &in.d[(i+ki)*W + j];
                const double* krow  = &ker.d[ki*K];
                for (size_t kj = 0; kj < K; ++kj)
                    acc += inrow[kj] * krow[kj];
            }
            out(i,j) = acc;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " H W K threads\n";
        return 1;
    }
    size_t H = std::stoul(argv[1]);
    size_t W = std::stoul(argv[2]);
    size_t K = std::stoul(argv[3]);
    unsigned threads = std::stoul(argv[4]);

    Img img(H, W), ker(K, K), out(H-K+1, W-K+1);
    fill_deterministic(img);
    fill_deterministic(ker);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> pool;
    size_t rows_per = out.h / threads;
    size_t start = 0;
    for (unsigned t = 0; t < threads; ++t) {
        size_t end = (t == threads-1) ? out.h : start + rows_per;
        pool.emplace_back(worker, std::cref(img), std::cref(ker), std::ref(out), start, end);
        start = end;
    }
    for (auto& th : pool) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Multi-thread conv (" << threads << " threads) took " << (secs*1000.0) << " ms\n";
    return 0;
}
