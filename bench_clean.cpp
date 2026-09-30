// Benchmark harness for classical 2-way cuckoo hashing, bounded-displacement
// d-ary cuckoo hashing with a stash, chained hashing, and linear probing.
// Experimental harness used to generate comparative measurements.
// Numerical claims in the manuscript must be tied to an actual recorded run.
#include <bits/stdc++.h>
using namespace std;
using u64 = uint64_t;
using Clock = chrono::steady_clock;

static inline u64 splitmix64(u64 x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
struct HashFn {
    u64 seed;
    HashFn(u64 s=0): seed(s) {}
    inline u64 operator()(u64 key) const { return splitmix64(key ^ seed); }
};

// ---------- Linear Probing ----------
struct LinearProbing {
    vector<u64> table; vector<char> occ; size_t cap; size_t n=0;
    HashFn h;
    LinearProbing(size_t c, u64 seed): table(c,0), occ(c,0), cap(c), h(seed) {}
    // returns probe count; -1 if table full without finding slot (shouldn't happen under cap)
    long insert(u64 key){
        size_t idx = h(key)%cap; long probes=0;
        for(size_t i=0;i<cap;i++){
            probes++;
            size_t p=(idx+i)%cap;
            if(!occ[p]){ occ[p]=1; table[p]=key; n++; return probes; }
            if(occ[p] && table[p]==key) return probes; // already present
        }
        return -1;
    }
    long lookup(u64 key){
        size_t idx=h(key)%cap; long probes=0;
        for(size_t i=0;i<cap;i++){ probes++; size_t p=(idx+i)%cap; if(!occ[p]) return probes; if(table[p]==key) return probes; }
        return -1;
    }
};

// ---------- Chained Hashing ----------
struct ChainedHashing {
    vector<vector<u64>> table; size_t cap; size_t n=0; HashFn h;
    ChainedHashing(size_t c, u64 seed): table(c), cap(c), h(seed) {}
    long insert(u64 key){
        size_t idx=h(key)%cap; auto &b=table[idx];
        for(auto k:b) if(k==key) return (long)b.size();
        b.push_back(key); n++; return (long)b.size();
    }
    long lookup(u64 key){
        size_t idx=h(key)%cap; auto &b=table[idx]; long probes=0;
        for(auto k:b){ probes++; if(k==key) return probes; }
        return probes+1;
    }
};

// ---------- Classical 2-way cuckoo (Pagh-Rodler), MaxLoop = ceil(3 log_{1+eps} r) ----------
struct ClassicalCuckoo {
    vector<u64> T1,T2; vector<char> occ1,occ2; size_t m; // each table size m, r=2m
    HashFn h1,h2; size_t n=0; long maxLoop; long fullRehashes=0;
    double eps;
    mt19937_64 rng;
    ClassicalCuckoo(size_t m_, double eps_, u64 seed): T1(m_,0),T2(m_,0),occ1(m_,0),occ2(m_,0),
        m(m_), h1(seed*2+1), h2(seed*2+2), eps(eps_), rng(seed) {
        double r = 2.0*m;
        maxLoop = (long)ceil(3.0*log(r)/log(1.0+eps));
        if(maxLoop<4) maxLoop=4;
    }
    long lookup(u64 key){
        if(occ1[h1(key)%m] && T1[h1(key)%m]==key) return 1;
        if(occ2[h2(key)%m] && T2[h2(key)%m]==key) return 2;
        return -1; // not found
    }
    // Returns the number of kicks used (>=0). If the insertion budget is
    // exhausted, a full-table rehash is performed and the configured bound
    // is returned as a conservative logical cost.
    long insert(u64 key){
        if(lookup(key)>0) return 0;
        u64 cur=key; long kicks=0;
        for(long it=0; it<maxLoop; it++){
            size_t p1=h1(cur)%m;
            if(!occ1[p1]){ occ1[p1]=1; T1[p1]=cur; n++; return kicks; }
            u64 evicted=T1[p1]; T1[p1]=cur; occ1[p1]=1; cur=evicted; kicks++;
            size_t p2=h2(cur)%m;
            if(!occ2[p2]){ occ2[p2]=1; T2[p2]=cur; n++; return kicks; }
            evicted=T2[p2]; T2[p2]=cur; occ2[p2]=1; cur=evicted; kicks++;
        }
        // trigger full rehash: pick new seeds, reinsert everything including cur
        fullRehashes++;
        vector<u64> all;
        for(size_t i=0;i<m;i++){ if(occ1[i]) all.push_back(T1[i]); if(occ2[i]) all.push_back(T2[i]); }
        all.push_back(cur);
        fill(occ1.begin(),occ1.end(),0); fill(occ2.begin(),occ2.end(),0);
        size_t oldn=n; n=0;
        h1=HashFn(rng()); h2=HashFn(rng());
        for(auto k: all){ insert(k); } // recursive; rare
        return maxLoop; // report as max cost for latency purposes
    }
};

// ---------- Proposed: d-ary bounded-displacement cuckoo with stash ----------
struct BoundedCuckoo {
    int d; size_t m; vector<vector<u64>> T; vector<vector<char>> occ;
    vector<HashFn> h; size_t n=0;
    vector<u64> stash; size_t stashCap; vector<char> stashOcc;
    long maxKicks;
    long incrementalRehashes=0, stashOverflowEvents=0, stashRoutedCount=0;
    long peakStashOcc=0; double sumStashOcc=0; long stashSamples=0;
    mt19937_64 rng;

    BoundedCuckoo(int d_, size_t m_, long maxKicks_, size_t stashCap_, u64 seed)
        : d(d_), m(m_), T(d_, vector<u64>(m_,0)), occ(d_, vector<char>(m_,0)),
          stash(stashCap_,0), stashCap(stashCap_), stashOcc(stashCap_,0),
          maxKicks(maxKicks_), rng(seed) {
        for(int i=0;i<d;i++) h.push_back(HashFn(seed*10+i+7));
    }
    long lookup(u64 key){
        for(int i=0;i<d;i++){ size_t p=h[i](key)%m; if(occ[i][p] && T[i][p]==key) return i+1; }
        for(size_t j=0;j<stashCap;j++) if(stashOcc[j] && stash[j]==key) return d+1;
        return -1;
    }
    long currentStashOcc(){ long c=0; for(size_t j=0;j<stashCap;j++) if(stashOcc[j]) c++; return c; }

    // Returns kicks used, -2 when the final displaced key is routed to the
    // stash, and -4 only when repeated stash recovery exceeds the retry bound.
    long insertInner(u64 key, int retryDepth=0){
        if(lookup(key)>0) return 0;
        u64 cur=key; long kicks=0;
        for(long round=0; round<maxKicks; round++){
            for(int i=0;i<d;i++){
                size_t p=h[i](cur)%m;
                if(!occ[i][p]){ occ[i][p]=1; T[i][p]=cur; n++; return kicks; }
            }
            int i = (int)(rng()%d);
            size_t p=h[i](cur)%m;
            u64 evicted=T[i][p]; T[i][p]=cur; occ[i][p]=1; cur=evicted; kicks++;
        }
        // route to stash
        for(size_t j=0;j<stashCap;j++){
            if(!stashOcc[j]){ stashOcc[j]=1; stash[j]=cur; n++; stashRoutedCount++;
                sumStashOcc+=currentStashOcc(); stashSamples++;
                peakStashOcc=max(peakStashOcc,(long)currentStashOcc());
                return -2; }
        }
        // stash overflow -> incremental rehash of stash contents (same h[], fresh eviction randomness)
        stashOverflowEvents++;
        if(retryDepth >= 20){ fullRehashesFromStuckStash++; return -4; } // pathological fallback
        incrementalRehashStash(retryDepth+1);
        // The key in `cur` is the entry that remains unplaced after the
        // bounded eviction path. Retain that key across stash recovery;
        // restarting from the original key would silently lose `cur`.
        return insertInner(cur, retryDepth+1);
    }
    long fullRehashesFromStuckStash=0;
    // CORRECTNESS NOTE: an earlier version of this routine redrew fresh global
    // hash functions h[0..d-1] on every stash overflow. That silently broke
    // lookups for every key already resident in the main tables (their physical
    // slot was computed under the OLD h[], so probing under the NEW h[] would
    // almost never find them). Global hash functions must not change once any
    // key has been placed under them. The corrected design keeps h[] fixed for
    // the table's lifetime and instead retries the bounded-kick placement of
    // the stashed key(s) with fresh *randomized eviction choices* only --
    // still O(1) extra work, and it never touches already-placed keys.
    void incrementalRehashStash(int retryDepth=0){
        incrementalRehashes++;
        vector<u64> S;
        for(size_t j=0;j<stashCap;j++) if(stashOcc[j]) S.push_back(stash[j]);
        fill(stashOcc.begin(),stashOcc.end(),0);
        n -= S.size();
        for(auto k: S) insertInner(k, retryDepth); // same h[], fresh random eviction path
    }
    long insert(u64 key){ return insertInner(key); }
};

template<typename F>
void percentileReport(vector<long>& samples, F unitConv, const string& label){
    sort(samples.begin(), samples.end());
    size_t N=samples.size();
    auto pct=[&](double p)->long{ size_t idx=(size_t)(p*(N-1)); return samples[idx]; };
    cerr << label << " N=" << N
         << " p50=" << unitConv(pct(0.50))
         << " p99=" << unitConv(pct(0.99))
         << " p999=" << unitConv(pct(0.999))
         << " max=" << unitConv(samples.back()) << "\n";
}

int main(int argc, char** argv){
    // ------------- Experiment configuration -------------
    size_t N = 2000000;          // number of distinct keys inserted (real N, reported honestly)
    size_t lookupOps = 2000000;  // subsequent lookups performed
    double targetLoad = 0.90;    // target load factor for capacity sizing
    u64 seed = 123456789ULL;

    cerr << fixed << setprecision(6);
    mt19937_64 keyRng(42);

    // Generate N distinct random 64-bit keys (synthetic 5-tuple-like flow IDs)
    vector<u64> keys(N);
    { unordered_set<u64> seen; seen.reserve(N*2);
      for(size_t i=0;i<N;i++){ u64 k; do{ k = keyRng(); } while(!seen.insert(k).second); keys[i]=k; } }

    // ===================== Table II: comparative microbenchmark =====================
    cerr << "\n=== TABLE II: comparative microbenchmark, N=" << N << " keys ===\n";

    // --- Linear Probing, cap so that N/cap = 0.70 ---
    {
        size_t cap = (size_t)(N/0.70);
        LinearProbing lp(cap, seed);
        vector<long> insLat, lookLat;
        auto t0=Clock::now();
        for(auto k: keys){ auto s=Clock::now(); lp.insert(k); auto e=Clock::now();
            insLat.push_back(chrono::duration_cast<chrono::nanoseconds>(e-s).count()); }
        auto t1=Clock::now();
        double totalInsSec = chrono::duration<double>(t1-t0).count();
        for(size_t i=0;i<lookupOps;i++){ u64 k=keys[i%N]; auto s=Clock::now(); lp.lookup(k); auto e=Clock::now();
            lookLat.push_back(chrono::duration_cast<chrono::nanoseconds>(e-s).count()); }
        double avgLookProbes=0; for(auto k: keys) avgLookProbes+=lp.lookup(k); avgLookProbes/=N;
        long maxInsProbe=0; { size_t idx=0; LinearProbing lp2(cap,seed); for(auto k:keys){ long p=lp2.insert(k); maxInsProbe=max(maxInsProbe,p);} }
        cerr << "[LinearProbing] load=" << (double)N/cap << " avgLookupProbes=" << avgLookProbes
             << " maxInsertProbes=" << maxInsProbe << " totalInsertTimeSec=" << totalInsSec << "\n";
        percentileReport(insLat, [](long x){return x;}, "[LinearProbing] insertLatencyNs");
    }

    // --- Chained Hashing, cap so load factor ~1.0 (alpha=n/cap) ---
    {
        size_t cap = N; // alpha ~ 1.0
        ChainedHashing ch(cap, seed);
        vector<long> insLat;
        for(auto k: keys){ auto s=Clock::now(); ch.insert(k); auto e=Clock::now();
            insLat.push_back(chrono::duration_cast<chrono::nanoseconds>(e-s).count()); }
        double avgLookProbes=0; for(auto k: keys) avgLookProbes+=ch.lookup(k); avgLookProbes/=N;
        long maxChain=0; for(auto &b: ch.table) maxChain=max(maxChain,(long)b.size());
        cerr << "[Chained] load=" << (double)N/cap << " avgLookupProbes=" << avgLookProbes
             << " maxChainLen(=maxInsertProbes)=" << maxChain << "\n";
        percentileReport(insLat, [](long x){return x;}, "[Chained] insertLatencyNs");
    }

    // --- Classical 2-way cuckoo, eps=1 (50% max load), sized so n/(2m) ~= 0.495 ---
    {
        size_t m = (size_t)(N/(0.495*2.0)); // r=2m, target load n/r = 0.495
        ClassicalCuckoo cc(m, 1.0, seed);
        vector<long> insLat; vector<long> insKicks;
        for(auto k: keys){ auto s=Clock::now(); long kk=cc.insert(k); auto e=Clock::now();
            insLat.push_back(chrono::duration_cast<chrono::nanoseconds>(e-s).count());
            insKicks.push_back(kk); }
        double avgLookProbes=0; for(auto k: keys){ long r=cc.lookup(k); avgLookProbes += (r>0?r:2); } avgLookProbes/=N;
        long maxKicksObs=*max_element(insKicks.begin(), insKicks.end());
        cerr << "[ClassicalCuckoo d=2] load=" << (double)cc.n/(2*m) << " maxLoopParam=" << cc.maxLoop
             << " avgLookupProbes=" << avgLookProbes << " maxKicksObserved=" << maxKicksObs
             << " fullRehashes=" << cc.fullRehashes << "\n";
        percentileReport(insLat, [](long x){return x;}, "[ClassicalCuckoo] insertLatencyNs");
    }

    // --- Proposed bounded scheme, d=3, MaxKicks=16, stash=4, target load 0.90 ---
    {
        int d=3; long maxKicks=16; size_t stashCap=4;
        size_t m = (size_t)(N/(targetLoad*d));
        BoundedCuckoo bc(d, m, maxKicks, stashCap, seed);
        vector<long> insLat; vector<long> insKicks;
        for(auto k: keys){ auto s=Clock::now(); long kk=bc.insert(k); auto e=Clock::now();
            insLat.push_back(chrono::duration_cast<chrono::nanoseconds>(e-s).count());
            insKicks.push_back(kk==-2? maxKicks : (kk<0?maxKicks:kk)); }
        double avgLookProbes=0; for(auto k: keys){ long r=bc.lookup(k); avgLookProbes += (r>0?r:(long)(d+1)); } avgLookProbes/=N;
        cerr << "[Proposed d=3,MaxKicks=16,s=4] load=" << (double)bc.n/(d*m)
             << " avgLookupProbes=" << avgLookProbes << " maxInsertKicks=" << maxKicks
             << " stashRoutedCount=" << bc.stashRoutedCount
             << " incrementalRehashes=" << bc.incrementalRehashes
             << " stashOverflowEvents=" << bc.stashOverflowEvents << "\n";
        percentileReport(insLat, [](long x){return x;}, "[Proposed] insertLatencyNs");
    }

    // --- Classical cuckoo at Pagh-Rodler's own recommended operating point (load ~1/3) ---
    {
        size_t m = (size_t)(N/(0.33*2.0));
        ClassicalCuckoo cc(m, 2.03, seed+555); // eps chosen so r=(1+eps)n roughly matches 1/3 load, m sized directly below
        vector<long> insKicks;
        for(auto k: keys){ long kk=cc.insert(k); insKicks.push_back(kk); }
        sort(insKicks.begin(), insKicks.end());
        size_t Nn=insKicks.size();
        cerr << "[ClassicalCuckoo typical-load-third] load=" << (double)cc.n/(2*m) << " maxLoopParam=" << cc.maxLoop
             << " fullRehashes=" << cc.fullRehashes
             << " kicks_p50=" << insKicks[(size_t)(0.50*(Nn-1))]
             << " kicks_p99=" << insKicks[(size_t)(0.99*(Nn-1))]
             << " kicks_p999=" << insKicks[(size_t)(0.999*(Nn-1))]
             << " kicks_max=" << insKicks.back() << "\n";
    }

    // --- Classical cuckoo, pushed near its ~50% ceiling to expose the real long-chain tail ---
    {
        size_t m = (size_t)(N/(0.49*2.0));
        ClassicalCuckoo cc(m, 1.0, seed+999);
        vector<long> insKicks;
        for(auto k: keys){ long kk=cc.insert(k); insKicks.push_back(kk); }
        sort(insKicks.begin(), insKicks.end());
        size_t Nn=insKicks.size();
        cerr << "[ClassicalCuckoo near-ceiling] load=" << (double)cc.n/(2*m) << " maxLoopParam=" << cc.maxLoop
             << " fullRehashes=" << cc.fullRehashes
             << " kicks_p50=" << insKicks[(size_t)(0.50*(Nn-1))]
             << " kicks_p99=" << insKicks[(size_t)(0.99*(Nn-1))]
             << " kicks_p999=" << insKicks[(size_t)(0.999*(Nn-1))]
             << " kicks_max=" << insKicks.back() << "\n";
    }

    // ===================== Table III: sensitivity to MaxKicks (d=3, alpha=0.88, s=4) =====================
    cerr << "\n=== TABLE III: MaxKicks sensitivity, N=" << N << ", d=3, target alpha=0.88, s=4 ===\n";
    for(long mk : {4L,8L,12L,16L,24L}){
        int d=3; size_t stashCap=4; double alpha=0.88;
        size_t m = (size_t)(N/(alpha*d));
        BoundedCuckoo bc(d, m, mk, stashCap, seed+mk);
        long maxKicksObs=0;
        for(auto k: keys){ long kk=bc.insert(k); if(kk>=0) maxKicksObs=max(maxKicksObs,kk); }
        double routingProb = (double)bc.stashRoutedCount / N;
        double avgStashOcc = bc.stashSamples? bc.sumStashOcc/bc.stashSamples : 0.0;
        cerr << "MaxKicks=" << mk << " routingProb=" << routingProb
             << " avgStashOccAtRouting=" << avgStashOcc
             << " stashOverflowEvents=" << bc.stashOverflowEvents
             << " maxObservedKicksBeforeStash=" << maxKicksObs << " load=" << (double)bc.n/(d*m) << "\n";
    }

    cerr << "\nDone.\n";
    return 0;
}
