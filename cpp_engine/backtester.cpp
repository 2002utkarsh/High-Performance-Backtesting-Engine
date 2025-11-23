#include "backtester.h"
#include <vector>
#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>

// -------------------------------------------------
// Legacy Portfolio (Internal)
// -------------------------------------------------
class Portfolio {
public:
    double initial_cash;
    double cash;
    int holdings;

    Portfolio(double initial_cash)
        : initial_cash(initial_cash), cash(initial_cash), holdings(0) {}

    void execute_buy(const StockTick& tick) {
        if (cash >= tick.close) {
            holdings += 1;
            cash -= tick.close;
        }
    }

    void execute_sell(const StockTick& tick) {
        if (holdings > 0) {
            holdings -= 1;
            cash += tick.close;
        }
    }

    double get_total_value(const StockTick& tick) const {
        return cash + holdings * tick.close;
    }
};

// -------------------------------------------------
// Engine Implementation
// -------------------------------------------------

Engine::Engine() : running(false) {}

Engine::~Engine() {
    stop();
}

void Engine::start() {
    if (running) return;
    running = true;
    worker_thread = std::thread(&Engine::loop, this);
}

void Engine::stop() {
    if (!running) return;
    running = false;
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
}

bool Engine::ingest_packet(const BinaryPacket& packet) {
    BinaryPacket* slot = ring_buffer.next_write_slot();
    if (slot) {
        // Zero-copy: In a real scenario, we might construct directly in place.
        // Here we copy into the pre-allocated slot.
        std::memcpy(slot, &packet, sizeof(BinaryPacket));
        ring_buffer.commit_write();
        return true;
    }
    return false; // Buffer full, drop packet
}

const std::vector<NormalizedRow>& Engine::get_normalized_data() const {
    return normalized_data;
}

size_t Engine::get_data_count_safe() {
    std::lock_guard<std::mutex> lock(data_mutex);
    return normalized_data.size();
}

void Engine::copy_data_safe(NormalizedRow* out_buffer, int max_count) {
    std::lock_guard<std::mutex> lock(data_mutex);
    if (!normalized_data.empty()) {
        size_t count = std::min((size_t)max_count, normalized_data.size());
        std::memcpy(out_buffer, normalized_data.data(), count * sizeof(NormalizedRow));
    }
}

void Engine::loop() {
    while (running) {
        const BinaryPacket* packet = ring_buffer.next_read_slot();
        if (packet) {
            // Process packet
            // Simulate normalization logic (e.g., parsing payload)
            NormalizedRow row;
            row.timestamp = packet->timestamp;
            row.price = 100.0 + (packet->seq_num % 10); // Dummy logic
            row.size = 1.0;
            std::strncpy(row.symbol, "BTC", 7);

            {
                std::lock_guard<std::mutex> lock(data_mutex);
                normalized_data.push_back(row);
            }

            ring_buffer.commit_read();
        } else {
            // Busy-wait yield or sleep for low latency vs CPU usage trade-off
            std::this_thread::yield(); 
        }
    }
}

// -------------------------------------------------
// C API Implementation
// -------------------------------------------------

extern "C" {

    void* engine_create() {
        Engine* engine = new Engine();
        engine->start();
        return engine;
    }

    void engine_destroy(void* engine_ptr) {
        if (engine_ptr) {
            Engine* engine = static_cast<Engine*>(engine_ptr);
            delete engine;
        }
    }

    void engine_ingest(void* engine_ptr, int count) {
        Engine* engine = static_cast<Engine*>(engine_ptr);
        if (!engine) return;

        // Simulate high-frequency burst
        for (int i = 0; i < count; ++i) {
            BinaryPacket pkt;
            pkt.seq_num = i;
            pkt.timestamp = 1620000000 + i;
            pkt.type = 'T';
            // pkt.payload is uninitialized for speed in this mock
            
            // Spin until we can push (backpressure)
            while (!engine->ingest_packet(pkt)) {
                std::this_thread::yield();
            }
        }
    }

    int engine_get_data_count(void* engine_ptr) {
        Engine* engine = static_cast<Engine*>(engine_ptr);
        if (!engine) return 0;
        return engine->get_data_count_safe();
    }

    void engine_get_data(void* engine_ptr, NormalizedRow* out_buffer, int max_count) {
        Engine* engine = static_cast<Engine*>(engine_ptr);
        if (!engine) return;
        engine->copy_data_safe(out_buffer, max_count);
    }

    // Legacy function preserved
    void perform_backtest(const StockTick* ticks, int num_ticks, const int* signals, double* portfolio_history) {
        Portfolio portfolio(10000.0);
        for (int i = 0; i < num_ticks; ++i) {
            const StockTick& current_tick = ticks[i];
            if (signals[i] == 1) {
                portfolio.execute_buy(current_tick);
            } else if (signals[i] == -1) {
                portfolio.execute_sell(current_tick);
            }
            portfolio_history[i] = portfolio.get_total_value(current_tick);
        }
    }
}
