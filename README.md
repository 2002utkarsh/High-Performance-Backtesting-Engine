# High-Performance Backtesting Engine

A high-speed, modular trading and backtesting engine built in **C++** and **Python**.  
This project combines low-latency performance with a fully automated trading pipeline — from data ingestion to live execution.

---

## Project Overview

This system is designed to efficiently **backtest and execute algorithmic trading strategies** on cryptocurrency markets.  
It provides a C++ engine optimized for high event throughput, and a Python automation layer for live trading and analytics.

- **Backtesting Engine:** Capable of processing **350,000+ events/second** over **1.6M historical data points**.
- **Trading Pipeline:** Automates screening, signal generation, and trade execution using real-time data from the **KuCoin API**.
- **Extensible Architecture:** Built to support multiple exchanges, strategies, and asset classes with minimal modification.

---

## Features

- **High-performance backtesting** engine written in C++ for low-latency computation  
- **Automated trading pipeline** that ingests, analyzes, and executes trades live  
- **Signal generation module** for strategy backtesting and live evaluation  
- **Modular design** enabling custom strategies and exchange integrations  
- **Data ingestion** from KuCoin and other APIs via Python scripts  
- **Logging and analytics hooks** for detailed trade and performance metrics  

---

## Tech Stack

| Layer | Technology |
|-------|-------------|
| Core Engine | C++17 |
| Scripting / Automation | Python 3.10+ |
| Data APIs | KuCoin REST / WebSocket |
| Build System | CMake |
| Data Storage | CSV / Local DB |
| Optional | Docker for environment setup |

---

## Installation & Setup

### Prerequisites
- `CMake >= 3.20`
- `Python >= 3.10`
- `g++` or `clang` with C++17 support
- KuCoin API credentials (for live trading)

### Clone the Repository
```bash
git clone https://github.com/<your-username>/high-performance-trading-engine.git
cd high-performance-trading-engine
