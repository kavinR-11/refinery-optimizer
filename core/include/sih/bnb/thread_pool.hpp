#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <memory>

namespace sih {
namespace bnb {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) : m_stop(false), m_active_tasks(0) {
        if (num_threads == 0) num_threads = 1;
        for (size_t i = 0; i < num_threads; ++i) {
            m_workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->m_queue_mutex);
                        this->m_cv.wait(lock, [this]() {
                            return this->m_stop || !this->m_tasks.empty();
                        });

                        if (this->m_stop && this->m_tasks.empty()) {
                            return;
                        }

                        task = std::move(this->m_tasks.front());
                        this->m_tasks.pop();
                        this->m_active_tasks++;
                    }

                    task();

                    {
                        std::unique_lock<std::mutex> lock(this->m_queue_mutex);
                        this->m_active_tasks--;
                        if (this->m_tasks.empty() && this->m_active_tasks == 0) {
                            this->m_wait_cv.notify_all();
                        }
                    }
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            if (m_stop) {
                throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
            }
            m_tasks.emplace([task]() { (*task)(); });
        }
        m_cv.notify_one();
        return res;
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(m_queue_mutex);
        m_wait_cv.wait(lock, [this]() {
            return m_tasks.empty() && m_active_tasks == 0;
        });
    }

    size_t size() const noexcept {
        return m_workers.size();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        for (std::thread& worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;

    std::mutex m_queue_mutex;
    std::condition_variable m_cv;
    std::condition_variable m_wait_cv;
    std::atomic<bool> m_stop;
    std::atomic<int64_t> m_active_tasks;
};

} // namespace bnb
} // namespace sih
