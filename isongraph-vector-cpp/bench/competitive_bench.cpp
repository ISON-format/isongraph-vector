/**
 * Loader and measurement harness for benchmarks/competitive/ (the cross-library comparison).
 *
 * Reads the same raw float32 vectors, queries and float64 ground truth the Python harness feeds the
 * other systems, so every system sees identical data. Bench-only: the library is not touched.
 *
 * Build (MSVC, from a vcvars64 prompt; header-only library):
 *   cl /std:c++17 /O2 /EHsc /MD /DNDEBUG [/arch:AVX2] /I include /I vendor/ison-graph-cpp
 *      bench/competitive_bench.cpp /Fe:competitive_bench.exe /link psapi.lib
 *
 * Run:
 *   competitive_bench --ds-dir D:/.../prep/text384 --ds text384 --n 100000 --exp core,mutation,saveload
 *                     --queries 2000 --label cpp --out result.json
 *   competitive_bench ... --exp filter
 *
 * `core` times the exact scan (the library's hash-map loop, there is no scan cache in this port) and
 * the opt-in IVF index (nprobe sweep, 3 timed runs per point).
 */
#define NOMINMAX
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "../include/isongraph_vector.hpp"

using namespace isongraph_vector;
using Clock = std::chrono::steady_clock;

namespace {

double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

std::string fnum(double v) {
    char b[64];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}
std::string farr(const std::vector<double>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + fnum(v[i]);
    return s + "]";
}

std::string mem_json() {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc);
    return "{\"rss_mb\":" + fnum(pmc.WorkingSetSize / 1048576.0) + ",\"peak_mb\":" + fnum(pmc.PeakWorkingSetSize / 1048576.0) + "}";
}

std::vector<float> read_f32(const std::string& path, size_t start, size_t count, size_t d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(2); }
    f.seekg(static_cast<std::streamoff>(start * d * 4));
    std::vector<float> v(count * d);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * d * 4));
    return v;
}
std::vector<int32_t> read_i32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(2); }
    size_t n = static_cast<size_t>(f.tellg()) / 4;
    f.seekg(0);
    std::vector<int32_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
    return v;
}
std::vector<uint8_t> read_u8(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(2); }
    size_t n = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
    return v;
}

double pct(std::vector<double> s, double p) {
    if (s.empty()) return 0;
    std::sort(s.begin(), s.end());
    return s[static_cast<size_t>(std::llround((s.size() - 1) * p))];
}
double median(const std::vector<double>& v) { return pct(v, 0.5); }

struct Ctx {
    std::string dir;
    size_t d = 0, n = 0, k = 10;
    std::vector<std::vector<float>> queries;
    std::vector<std::vector<float>> base(size_t start, size_t count) const {
        auto flat = read_f32(dir + "/base.f32", start, count, d);
        std::vector<std::vector<float>> rows(count);
        for (size_t i = 0; i < count; ++i) rows[i].assign(flat.begin() + i * d, flat.begin() + (i + 1) * d);
        return rows;
    }
    std::vector<std::vector<int32_t>> gt(const std::string& kind) const {
        auto flat = read_i32(dir + "/gt_" + kind + std::to_string(n) + ".i32");
        std::vector<std::vector<int32_t>> rows(flat.size() / 100);
        for (size_t i = 0; i < rows.size(); ++i) rows[i].assign(flat.begin() + i * 100, flat.begin() + (i + 1) * 100);
        return rows;
    }
};

double recall(const std::vector<std::vector<size_t>>& found, const std::vector<std::vector<int32_t>>& truth, size_t k) {
    size_t hit = 0;
    for (size_t i = 0; i < found.size(); ++i) {
        std::unordered_set<int32_t> want(truth[i].begin(), truth[i].begin() + k);
        for (size_t j = 0; j < std::min(k, found[i].size()); ++j) hit += want.count(static_cast<int32_t>(found[i][j]));
    }
    return static_cast<double>(hit) / (k * std::max<size_t>(found.size(), 1));
}

enum class Mode { Exact, Ann };

std::vector<size_t> do_search(const EmbeddingStore& s, const std::vector<float>& q, size_t k, Mode p, const std::string& ty) {
    auto r = s.similaritySearchVector(q, k, ty, 0.0f, p == Mode::Exact);
    std::vector<size_t> ids;
    ids.reserve(r.size());
    for (auto& x : r) ids.push_back(static_cast<size_t>(std::stoull(x.node_ref.id)));
    return ids;
}

void run_loop(const EmbeddingStore& s, const std::vector<std::vector<float>>& qs, size_t nq, size_t k, Mode p, const std::string& ty,
              std::vector<double>& lat, std::vector<std::vector<size_t>>* out) {
    for (size_t i = 0; i < nq; ++i) {
        double t = now_s();
        auto r = do_search(s, qs[i], k, p, ty);
        lat.push_back(now_s() - t);
        if (out) out->push_back(std::move(r));
    }
}

/** warm up, size the query count to `budget` seconds, then 3 timed runs. Returns a JSON object body (no braces). */
std::string point_impl(const EmbeddingStore& s, const Ctx& c, const std::vector<std::vector<int32_t>>& truth, Mode p, const std::string& ty,
                  double budget, double* qps_out, double* rec_out) {
    size_t nqall = c.queries.size();
    {
        double w0 = now_s();
        for (size_t i = 0; i < std::min<size_t>(50, nqall); ++i) {
            do_search(s, c.queries[i], c.k, p, ty);
            if (now_s() - w0 > 3.0) break;
        }
    }
    std::vector<double> probe;
    run_loop(s, c.queries, std::min<size_t>(5, nqall), c.k, p, ty, probe, nullptr);
    double est = std::max(median(probe), 1e-7);
    size_t nq = std::min(nqall, std::max<size_t>(30, static_cast<size_t>(budget / est)));
    std::vector<double> runs, lats;
    double rec = 0;
    for (int r = 0; r < 3; ++r) {
        std::vector<double> lat;
        std::vector<std::vector<size_t>> res;
        run_loop(s, c.queries, nq, c.k, p, ty, lat, r == 0 ? &res : nullptr);
        if (r == 0) rec = recall(res, truth, c.k);
        double total = 0;
        for (double x : lat) total += x;
        runs.push_back(nq / total);
        lats.insert(lats.end(), lat.begin(), lat.end());
        if (total > budget * 1.5) break;
    }
    double mean = 0;
    for (double x : lats) mean += x;
    mean = mean / std::max<size_t>(lats.size(), 1);
    double qps = median(runs);
    if (qps_out) *qps_out = qps;
    if (rec_out) *rec_out = rec;
    std::ostringstream o;
    o << "\"nq\":" << nq << ",\"recall\":" << fnum(rec) << ",\"qps_1t\":" << fnum(qps) << ",\"qps_1t_min\":" << fnum(*std::min_element(runs.begin(), runs.end()))
      << ",\"qps_1t_max\":" << fnum(*std::max_element(runs.begin(), runs.end())) << ",\"qps_1t_runs\":" << farr(runs)
      << ",\"median_ms\":" << fnum(pct(lats, 0.5) * 1000) << ",\"p95_ms\":" << fnum(pct(lats, 0.95) * 1000) << ",\"p99_ms\":" << fnum(pct(lats, 0.99) * 1000)
      << ",\"mean_ms\":" << fnum(mean * 1000);
    return o.str();
}

/** The batched call over all hardware threads: warm up once, then 3 timed runs of the whole query set. Returns a JSON object. */
std::string batch_point(const EmbeddingStore& s, const Ctx& c, const std::vector<std::vector<int32_t>>& truth, Mode p) {
    auto run = [&](double* qps) {
        double t = now_s();
        auto r = s.similaritySearchBatch(c.queries, c.k, "", 0.0f, p == Mode::Exact, 0);
        *qps = c.queries.size() / (now_s() - t);
        return r;
    };
    double q0 = 0;
    run(&q0);
    std::vector<double> qps;
    std::vector<std::vector<size_t>> found;
    for (int i = 0; i < 3; ++i) {
        double q = 0;
        auto r = run(&q);
        qps.push_back(q);
        if (i == 2) {
            for (auto& row : r) {
                std::vector<size_t> ids;
                for (auto& x : row) ids.push_back(static_cast<size_t>(std::stoull(x.node_ref.id)));
                found.push_back(std::move(ids));
            }
        }
    }
    std::ostringstream o;
    o << "{\"nq\":" << c.queries.size() << ",\"threads\":" << std::thread::hardware_concurrency() << ",\"qps\":" << fnum(median(qps))
      << ",\"qps_runs\":" << farr(qps) << ",\"recall\":" << fnum(recall(found, truth, c.k)) << "}";
    return o.str();
}

std::vector<size_t> hnsw_knobs() { return {10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 400, 640}; }

size_t nlist_for(size_t n) { return static_cast<size_t>(std::min(std::max(std::nearbyint(4.0 * std::sqrt(static_cast<double>(n))), 16.0), 65536.0)); }
std::vector<size_t> knobs(size_t n) {
    size_t nl = nlist_for(n);
    std::vector<size_t> v;
    for (size_t x : {1, 2, 4, 8, 16, 32, 64, 128, 256}) if (x < nl) v.push_back(x);
    if (n <= 200000) v.push_back(nl);  // nprobe == nlist is the exact scan; measured separately at scale
    return v;
}

double ingest(EmbeddingStore& s, const Ctx& c, size_t n, const std::vector<uint8_t>* grp) {
    double secs = 0;
    static const char* names[] = {"t10", "t01", "rest"};
    for (size_t at = 0; at < n; at += 20000) {
        size_t m = std::min<size_t>(20000, n - at);
        auto rows = c.base(at, m);
        double t = now_s();
        for (size_t i = 0; i < m; ++i) {
            const char* ty = grp ? names[(*grp)[at + i]] : "doc";
            s.add(NodeRef(ty, std::to_string(at + i)), "", rows[i]);
        }
        secs += now_s() - t;
    }
    return secs;
}

/** Pin the process to one P-core (logical CPU 8 on the i9-14900HX; 0-15 are P-cores) around a single-threaded loop.
 *  Unpinned single-thread loops varied up to 3x between processes on this hybrid CPU. */
struct Pin {
    DWORD_PTR old = 0;
    Pin() {
        DWORD_PTR sys = 0;
        GetProcessAffinityMask(GetCurrentProcess(), &old, &sys);
        const char* e = std::getenv("BENCH_PIN_CPU");
        int cpu = e ? std::atoi(e) : 8;
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        SetProcessAffinityMask(GetCurrentProcess(), static_cast<DWORD_PTR>(1) << cpu);
    }
    ~Pin() { SetProcessAffinityMask(GetCurrentProcess(), old); }
};

std::string point(const EmbeddingStore& s, const Ctx& c, const std::vector<std::vector<int32_t>>& truth, Mode p, const std::string& ty,
                  double budget, double* qps_out = nullptr, double* rec_out = nullptr) {
    Pin pin;
    return point_impl(s, c, truth, p, ty, budget, qps_out, rec_out);
}

std::string arg(int argc, char** argv, const std::string& name, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) if (name == argv[i]) return argv[i + 1];
    return def;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = arg(argc, argv, "--ds-dir", ""), ds = arg(argc, argv, "--ds", ""), label = arg(argc, argv, "--label", "cpp");
    size_t n = std::stoull(arg(argc, argv, "--n", "100000")), nq = std::stoull(arg(argc, argv, "--queries", "1000"));
    double budget = std::stod(arg(argc, argv, "--budget", "15"));
    std::string exps = "," + arg(argc, argv, "--exp", "core") + ",";
    std::string out_path = arg(argc, argv, "--out", "");
    auto has = [&](const char* e) { return exps.find(std::string(",") + e + ",") != std::string::npos; };

    std::string meta;
    { std::ifstream f(dir + "/meta.json"); std::ostringstream o; o << f.rdbuf(); meta = o.str(); }
    auto pos = meta.find("\"d\":");
    Ctx c;
    c.dir = dir;
    c.d = std::stoull(meta.substr(pos + 4));
    c.n = n;
    auto qflat = read_f32(dir + "/queries.f32", 0, nq, c.d);
    for (size_t i = 0; i < nq; ++i) c.queries.emplace_back(qflat.begin() + i * c.d, qflat.begin() + (i + 1) * c.d);
    unsigned cpus = std::thread::hardware_concurrency();
    std::vector<std::string> runs;
    std::string mem_base = mem_json();
    std::cout << label << ": ds=" << ds << " n=" << n << " d=" << c.d << " queries=" << nq << " cpus=" << cpus << std::endl;

    auto header = [&](const std::string& sys) {
        std::ostringstream o;
        o << "{\"system\":\"" << sys << "\",\"ds\":\"" << ds << "\",\"n\":" << n << ",\"d\":" << c.d << ",\"k\":10,\"queries\":" << nq
          << ",\"all_threads\":" << cpus << ",\"version\":\"isongraph-vector-cpp (repo HEAD, MSVC /O2)\"";
        return o.str();
    };

    if (has("core") || has("mutation") || has("saveload")) {
        EmbeddingStore store;
        double ingest_s = ingest(store, c, n, nullptr);
        std::string mem_ing = mem_json();
        std::cout << "  ingest " << ingest_s << " s" << std::endl;
        auto truth = c.gt("");
        // ---- exact (hash-map scan; there is no cache in this port) ----
        double t0 = now_s();
        store.similaritySearchVector(c.queries[0], c.k, "", 0.0f, true);
        double first_ms = (now_s() - t0) * 1000;
        std::ostringstream ex;
        ex << header(label + "_exact") << ",\"build\":{\"ingest_s\":" << fnum(ingest_s) << ",\"index_s\":0,\"total_s\":" << fnum(ingest_s)
           << ",\"first_query_ms\":" << fnum(first_ms) << ",\"ready_s\":" << fnum(ingest_s + first_ms / 1000) << "},\"mem_baseline\":" << mem_base
           << ",\"mem_after_ingest\":" << mem_ing << ",\"mem_after_build\":" << mem_json() << ",\"mem_after_first_query\":" << mem_json();
        double qps_ex = 0;
        std::string p1 = point(store, c, truth, Mode::Exact, "", budget, &qps_ex);
        std::cout << "  exact qps=" << qps_ex << std::endl;
        ex << ",\"sweep\":[{" << p1 << ",\"knob\":null}],\"batch_allthreads\":" << batch_point(store, c, truth, Mode::Exact)
           << ",\"mem_after_sweep\":" << mem_json() << "}";
        runs.push_back(ex.str());

        // ---- ivf ----
        AnnOptions opt;
        opt.minSize = 0;
        store.enableAnn(opt);
        double t1 = now_s();
        AnnStats st = store.buildAnnIndex();
        double idx_s = now_s() - t1;
        std::cout << "  ivf build " << idx_s << " s nlist=" << st.nlist << std::endl;
        std::ostringstream iv;
        iv << header(label + "_ivf") << ",\"knob_name\":\"nprobe\",\"build\":{\"ingest_s\":" << fnum(ingest_s) << ",\"index_s\":" << fnum(idx_s)
           << ",\"total_s\":" << fnum(ingest_s + idx_s) << ",\"first_query_ms\":0,\"ready_s\":" << fnum(ingest_s + idx_s) << "},\"mem_baseline\":" << mem_base
           << ",\"mem_after_ingest\":" << mem_ing << ",\"mem_after_build\":" << mem_json() << ",\"index_stats\":{\"nlist\":" << st.nlist
           << ",\"trained_n\":" << st.trainedN << ",\"index_bytes\":" << st.indexBytes << ",\"vector_bytes\":" << st.vectorBytes << ",\"build_seconds\":" << fnum(st.buildSeconds) << "}";
        std::vector<std::string> pts;
        long ref_knob = -1;
        double ref_recall = 0;
        for (size_t kn : knobs(n)) {
            store.setAnnNprobe(kn);
            double qps = 0, rec = 0;
            std::string p = point(store, c, truth, Mode::Ann, "", budget, &qps, &rec);
            std::cout << "  ivf nprobe=" << kn << " recall=" << rec << " qps1t=" << qps << std::endl;
            pts.push_back("{" + p + ",\"knob\":" + std::to_string(kn) + ",\"batch_allthreads\":" + batch_point(store, c, truth, Mode::Ann) + "}");
            if (ref_knob < 0 && rec >= 0.95) { ref_knob = static_cast<long>(kn); ref_recall = rec; }
        }
        if (ref_knob < 0) ref_knob = static_cast<long>(knobs(n).back());
        iv << ",\"sweep\":[";
        for (size_t i = 0; i < pts.size(); ++i) iv << (i ? "," : "") << pts[i];
        iv << "],\"ref_knob\":" << ref_knob << ",\"ref_recall_pre\":" << fnum(ref_recall) << ",\"mem_after_sweep\":" << mem_json();
        // concurrent clients: searches take turns on the index; measured not assumed
        {
            store.setAnnNprobe(static_cast<size_t>(ref_knob));
            unsigned conc = std::min(8u, cpus);
            size_t total = std::min<size_t>(nq, 400);
            double t = now_s();
            std::vector<std::thread> th;
            for (unsigned w = 0; w < conc; ++w)
                th.emplace_back([&, w] { for (size_t i = w; i < total; i += conc) do_search(store, c.queries[i], c.k, Mode::Ann, ""); });
            for (auto& x : th) x.join();
            iv << ",\"concurrent_clients\":{\"clients\":" << conc << ",\"qps\":" << fnum(total / (now_s() - t)) << "}";
        }

        // ---- mutation ----
        if (has("mutation")) {
            std::string gtm = dir + "/gt_mut_" + std::to_string(n) + ".i32";
            if (std::ifstream(gtm).good()) {
                auto upd = read_i32(dir + "/mut_" + std::to_string(n) + "_upd.i32");
                auto del = read_i32(dir + "/mut_" + std::to_string(n) + "_del.i32");
                auto flat = read_i32(gtm);
                std::vector<std::vector<int32_t>> truth_m(std::min(nq, flat.size() / 100));
                for (size_t i = 0; i < truth_m.size(); ++i) truth_m[i].assign(flat.begin() + i * 100, flat.begin() + (i + 1) * 100);
                store.setAnnNprobe(static_cast<size_t>(ref_knob));
                iv << ",\"mutation\":{\"ref_knob\":" << ref_knob << ",\"recall_before\":" << fnum(ref_recall);
                auto ins = c.base(n, 10000);
                double t = now_s();
                for (size_t i = 0; i < ins.size(); ++i) store.add(NodeRef("doc", std::to_string(n + i)), "", ins[i]);
                double dt = now_s() - t;
                iv << ",\"insert_10k\":{\"seconds\":" << fnum(dt) << ",\"per_s\":" << fnum(10000 / dt) << "}";
                auto nv = c.base(n + 10000, 10000);
                t = now_s();
                for (size_t j = 0; j < nv.size(); ++j) store.add(NodeRef("doc", std::to_string(upd[j])), "", nv[j]);
                dt = now_s() - t;
                iv << ",\"update_10k\":{\"seconds\":" << fnum(dt) << ",\"per_s\":" << fnum(10000 / dt) << "}";
                t = now_s();
                for (int32_t id : del) store.remove(NodeRef("doc", std::to_string(id)));
                dt = now_s() - t;
                iv << ",\"delete_10k\":{\"seconds\":" << fnum(dt) << ",\"per_s\":" << fnum(10000 / dt) << "}";
                t = now_s();
                do_search(store, c.queries[0], c.k, Mode::Ann, "");
                iv << ",\"first_query_after_ms\":" << fnum((now_s() - t) * 1000);
                double qps = 0, rec = 0;
                std::string p = point(store, c, truth_m, Mode::Ann, "", budget, &qps, &rec);
                AnnStats s2 = store.annStats();
                iv << ",\"recall_after\":" << fnum(rec) << ",\"after\":{" << p << "},\"index_stats_after\":{\"pending\":" << s2.pending << ",\"churn\":" << s2.churn
                   << ",\"trained_n\":" << s2.trainedN << ",\"builds\":" << s2.builds << "},\"mem_after\":" << mem_json() << "}";
                std::cout << "  mutation recall_after=" << rec << std::endl;
            } else {
                iv << ",\"mutation\":{\"skipped\":\"dataset has no spare rows for this n\"}";
            }
        }

        // ---- save / load (JSON sidecar) ----
        if (has("saveload")) {
            std::string path = dir + "/../../work/cpp_" + ds + "_" + std::to_string(n) + ".json";
            CreateDirectoryA((dir + "/../../work").c_str(), nullptr);
            iv << ",\"saveload\":{";
            try {
                double t = now_s();
                store.saveEmbeddings(path);
                double save_s = now_s() - t;
                std::ifstream f(path, std::ios::binary | std::ios::ate);
                double mb = static_cast<double>(f.tellg()) / 1048576.0;
                f.close();
                EmbeddingStore s2;
                t = now_s();
                s2.loadEmbeddings(path);
                double load_s = now_s() - t;
                AnnOptions o2;
                o2.minSize = 0;
                o2.nprobe = static_cast<size_t>(ref_knob);
                s2.enableAnn(o2);
                t = now_s();
                do_search(s2, c.queries[0], c.k, Mode::Ann, "");
                double fq = (now_s() - t) * 1000;
                iv << "\"save_s\":" << fnum(save_s) << ",\"disk_mb\":" << fnum(mb) << ",\"load_s\":" << fnum(load_s) << ",\"first_query_after_load_ms\":" << fnum(fq)
                   << ",\"time_to_ready_s\":" << fnum(load_s + fq / 1000) << ",\"note\":\"load = JSON parse + insert; first query builds the IVF index (never persisted)\"";
                // the opt-in binary snapshot of the same store (DESIGN.md section 11.2)
                std::string snap = dir + "/../../work/cpp_" + ds + "_" + std::to_string(n) + ".snap";
                t = now_s();
                store.saveSnapshot(snap);
                double ss = now_s() - t;
                std::ifstream f2(snap, std::ios::binary | std::ios::ate);
                double smb = static_cast<double>(f2.tellg()) / 1048576.0;
                f2.close();
                EmbeddingStore s3;
                t = now_s();
                s3.loadSnapshot(snap);
                double sl_s = now_s() - t;
                iv << ",\"snap_save_s\":" << fnum(ss) << ",\"snap_mb\":" << fnum(smb) << ",\"snap_load_s\":" << fnum(sl_s);
                std::remove(snap.c_str());
                std::remove(path.c_str());
            } catch (const std::exception& e) {
                iv << "\"error\":\"" << e.what() << "\"";
            }
            iv << "}";
        }
        iv << "}";
        runs.push_back(iv.str());
    }

    if (has("hnsw")) {
        EmbeddingStore store;
        double ingest_s = ingest(store, c, n, nullptr);
        std::string mem_ing = mem_json();
        std::cout << "  hnsw: ingest " << ingest_s << " s" << std::endl;
        auto truth = c.gt("");
        HnswOptions ho;
        ho.m = 16;
        ho.efConstruction = 200;
        AnnOptions opt = AnnOptions::makeHnsw(ho);
        opt.minSize = 0;
        store.enableAnn(opt);
        double t1 = now_s();
        AnnStats st = store.buildAnnIndex();
        double idx_s = now_s() - t1;
        std::string mem_build = mem_json();
        std::cout << "  hnsw build " << idx_s << " s graphNodes=" << st.graphNodes << " maxLevel=" << st.maxLevel << " indexBytes=" << st.indexBytes << std::endl;
        std::ostringstream hn;
        hn << header(label + "_hnsw") << ",\"knob_name\":\"ef_search\",\"build\":{\"ingest_s\":" << fnum(ingest_s) << ",\"index_s\":" << fnum(idx_s)
           << ",\"total_s\":" << fnum(ingest_s + idx_s) << ",\"first_query_ms\":0,\"ready_s\":" << fnum(ingest_s + idx_s) << "},\"mem_baseline\":" << mem_base
           << ",\"mem_after_ingest\":" << mem_ing << ",\"mem_after_build\":" << mem_build << ",\"index_stats\":{\"graph_nodes\":" << st.graphNodes
           << ",\"max_level\":" << st.maxLevel << ",\"index_bytes\":" << st.indexBytes << ",\"build_seconds\":" << fnum(st.buildSeconds)
           << ",\"default_ef_search\":" << st.efSearch << "}";
        std::vector<std::string> pts;
        long ref_knob = -1;
        double ref_recall = 0;
        std::vector<double> recs;
        for (size_t kn : hnsw_knobs()) {
            store.setAnnEfSearch(kn);
            double qps = 0, rec = 0;
            std::string p = point(store, c, truth, Mode::Ann, "", budget, &qps, &rec);
            std::string b = batch_point(store, c, truth, Mode::Ann);
            std::cout << "  hnsw ef=" << kn << " recall=" << rec << " qps1t=" << qps << std::endl;
            pts.push_back("{" + p + ",\"knob\":" + std::to_string(kn) + ",\"batch_allthreads\":" + b + "}");
            if (ref_knob < 0 && rec >= 0.95) { ref_knob = static_cast<long>(kn); ref_recall = rec; }
            recs.push_back(rec);
            if (recs.size() > 2 && rec >= 0.9995 && recs[recs.size() - 2] >= 0.9995) break;
        }
        if (ref_knob < 0) ref_knob = static_cast<long>(hnsw_knobs().back());
        hn << ",\"sweep\":[";
        for (size_t i = 0; i < pts.size(); ++i) hn << (i ? "," : "") << pts[i];
        hn << "],\"ref_knob\":" << ref_knob << ",\"ref_recall_pre\":" << fnum(ref_recall) << ",\"mem_after_sweep\":" << mem_json();
        // the library's automatic ef (what a user who just enables HNSW gets)
        store.setAnnEfSearch(0);
        {
            double qps = 0, rec = 0;
            std::string p = point(store, c, truth, Mode::Ann, "", budget, &qps, &rec);
            std::string b = batch_point(store, c, truth, Mode::Ann);
            hn << ",\"default_ef_point\":{" << p << ",\"knob\":" << st.efSearch << ",\"batch_allthreads\":" << b << "}";
            std::cout << "  hnsw default ef=" << st.efSearch << " recall=" << rec << " qps1t=" << qps << std::endl;
        }
        // concurrent clients (const search is thread-safe)
        store.setAnnEfSearch(static_cast<size_t>(ref_knob));
        hn << ",\"concurrent_clients\":[";
        bool firstc = true;
        for (unsigned conc : {8u, std::min(16u, cpus)}) {
            size_t total = std::min<size_t>(nq, 1000);
            double t = now_s();
            std::vector<std::thread> th;
            for (unsigned w = 0; w < conc; ++w)
                th.emplace_back([&, w] { for (size_t i = w; i < total; i += conc) do_search(store, c.queries[i], c.k, Mode::Ann, ""); });
            for (auto& x : th) x.join();
            hn << (firstc ? "" : ",") << "{\"clients\":" << conc << ",\"qps\":" << fnum(total / (now_s() - t)) << "}";
            firstc = false;
        }
        hn << "]}";
        runs.push_back(hn.str());
    }

    if (has("filter")) {
        auto g = read_u8(dir + "/grp_" + std::to_string(n) + ".u8");
        EmbeddingStore store;
        double ingest_s = ingest(store, c, n, &g);
        AnnOptions opt;
        opt.minSize = 0;
        store.enableAnn(opt);
        double t = now_s();
        store.buildAnnIndex();
        double idx_s = now_s() - t;
        std::ostringstream o;
        o << header(label + "_ivf") << ",\"filter\":{\"build_with_metadata_s\":" << fnum(ingest_s + idx_s) << ",\"mem\":" << mem_json() << ",\"subsets\":{";
        auto kn_all = knobs(n);
        std::set<size_t> idxs = {1, kn_all.size() / 3, kn_all.size() / 2, 2 * kn_all.size() / 3, kn_all.size() - 1};
        bool firstsub = true;
        for (auto [ty, lab] : {std::pair<const char*, const char*>{"t10", "f10"}, {"t01", "f01"}}) {
            auto truth = c.gt(std::string(lab) + "_");
            o << (firstsub ? "" : ",") << "\"" << lab << "\":[";
            firstsub = false;
            bool first = true;
            for (size_t i : idxs) {
                store.setAnnNprobe(kn_all[i]);
                double qps = 0, rec = 0;
                std::string p = point(store, c, truth, Mode::Ann, ty, budget, &qps, &rec);
                std::cout << "  filter " << lab << " nprobe=" << kn_all[i] << " recall=" << rec << " qps=" << qps << std::endl;
                o << (first ? "" : ",") << "{" << p << ",\"knob\":" << kn_all[i] << "}";
                first = false;
            }
            std::string pe = point(store, c, truth, Mode::Exact, ty, budget);
            o << ",{" << pe << ",\"knob\":\"exact\"}]";
        }
        o << "}}}";
        runs.push_back(o.str());
    }

    if (!out_path.empty()) {
        std::ofstream f(out_path);
        f << "{\"status\":\"ok\",\"label\":\"" << label << "\",\"runs\":[";
        for (size_t i = 0; i < runs.size(); ++i) f << (i ? "," : "") << runs[i];
        f << "]}";
        std::cout << "wrote " << out_path << std::endl;
    }
    return 0;
}
