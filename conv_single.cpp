#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
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

static void conv2d_valid(const Img& in, const Img& ker, Img& out) {
    const size_t H = in.h, W = in.w, K = ker.h;
    const size_t Ho = H - K + 1, Wo = W - K + 1;
    for (size_t i = 0; i < Ho; ++i) {
        for (size_t j = 0; j < Wo; ++j) {
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
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " H W K\n";
        return 1;
    }
    size_t H = std::stoul(argv[1]);
    size_t W = std::stoul(argv[2]);
    size_t K = std::stoul(argv[3]);

    Img img(H, W), ker(K, K), out(H-K+1, W-K+1);
    fill_deterministic(img);
    fill_deterministic(ker);

    auto t0 = std::chrono::high_resolution_clock::now();
    conv2d_valid(img, ker, out);
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Single-thread conv took " << (secs*1000.0) << " ms\n";
    return 0;
}
