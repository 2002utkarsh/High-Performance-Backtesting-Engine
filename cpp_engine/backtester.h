#ifndef BACKTESTER_H
#define BACKTESTER_H

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <functional>

// -------------------------------------------------
// Binary Protocol & Data Structures
// -------------------------------------------------

// Compact binary packet for high-frequency ingestion
struct BinaryPacket {
    uint32_t seq_num;
    uint64_t timestamp;
    char type; // 'T' for trade, 'Q' for quote
    char payload[55]; // Padding to make struct 64 bytes (cache-line friendly)
};

// Represents a normalized row for Parquet export
struct NormalizedRow {
    uint64_t timestamp;
    double price;
    double size;
    char symbol[8];
};

// -------------------------------------------------
// Zero-Copy Ring Buffer
// -------------------------------------------------

// Fixed-size circular buffer for lock-free(ish) producer-consumer
class RingBuffer {
public:
    static const size_t SIZE = 1024 * 16; // Power of 2
    BinaryPacket buffer[SIZE];
    std::atomic<size_t> head{0}; // Write index
    std::atomic<size_t> tail{0}; // Read index

    // Returns pointer to next write slot or nullptr if full
    BinaryPacket* next_write_slot() {
        size_t current_head = head.load(std::memory_order_relaxed);
        size_t next_head = (current_head + 1) % SIZE;
        if (next_head == tail.load(std::memory_order_acquire)) {
            return nullptr; // Full
        }
        return &buffer[current_head];
    }

    void commit_write() {
        head.store((head.load(std::memory_order_relaxed) + 1) % SIZE, std::memory_order_release);
    }

    // Returns pointer to next read slot or nullptr if empty
    const BinaryPacket* next_read_slot() {
        size_t current_tail = tail.load(std::memory_order_relaxed);
        if (current_tail == head.load(std::memory_order_acquire)) {
            return nullptr; // Empty
        }
        return &buffer[current_tail];
    }

    void commit_read() {
        tail.store((tail.load(std::memory_order_relaxed) + 1) % SIZE, std::memory_order_release);
    }
};

// -------------------------------------------------
// Event Loop Engine
// -------------------------------------------------

class Engine {
public:
    Engine();
    ~Engine();

    void start();
    void stop();
    
    // Ingests a raw packet (Producer)
    bool ingest_packet(const BinaryPacket& packet);

    // Returns normalized data for Python to pick up
    const std::vector<NormalizedRow>& get_normalized_data() const;

    // Thread-safe accessors
    size_t get_data_count_safe();
    void copy_data_safe(NormalizedRow* out_buffer, int max_count);

private:
    void loop(); // Consumer thread function

    RingBuffer ring_buffer;
    std::vector<NormalizedRow> normalized_data;
    
    std::thread worker_thread;
    std::atomic<bool> running;
    std::mutex data_mutex; // Protects normalized_data access
};

// -------------------------------------------------
// Legacy Structures (Kept for compatibility if needed, or refactored)
// -------------------------------------------------

struct StockTick {
    long long timestamp;
    double open;
    double high;
    double low;
    double close;
    int volume;
};

// -------------------------------------------------
// C API for Python
// -------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

// Initialize and start the engine
void* engine_create();
void engine_destroy(void* engine_ptr);

// Ingest data (simulated high-frequency stream)
void engine_ingest(void* engine_ptr, int count);

// Retrieve normalized data count
int engine_get_data_count(void* engine_ptr);

// Retrieve normalized data (copy to buffer)
void engine_get_data(void* engine_ptr, NormalizedRow* out_buffer, int max_count);

// Legacy backtest function
void perform_backtest(const StockTick* ticks, int num_ticks, const int* signals, double* portfolio_history);

#ifdef __cplusplus
}
#endif

#endif // BACKTESTER_H
