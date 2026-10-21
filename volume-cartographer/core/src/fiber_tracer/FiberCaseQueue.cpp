#if defined(__APPLE__) && !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 700
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
// ucontext requires X/Open declarations, but guarded anonymous stacks also
// need Darwin extensions hidden by the corresponding strict POSIX mode.
#define _DARWIN_C_SOURCE 1
#endif
#include "vc/fiber_tracer/FiberCaseQueue.hpp"
#include <atomic>
#include <algorithm>
#include <utility>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <ucontext.h>
#include <sys/mman.h>
#include <unistd.h>
#define VC_CASE_CONTEXT_SUPPORTED 1
#endif

namespace vc::fiber_tracer {
namespace {
std::mutex wakeMutex;
std::condition_variable wake;
std::atomic<size_t> epoch{0};
thread_local FiberConstructionCpuPermit* outsidePermit = nullptr;
}

#ifdef VC_CASE_CONTEXT_SUPPORTED
class FiberCaseQueue::Impl {
public:
    struct Task {
        ucontext_t context{};
        void* allocation = MAP_FAILED;
        size_t bytes = 0, page = 0;
        bool done = true, child = false, interruptible = true, cancelledWake = false;
        std::function<void()> work;
        std::function<bool()> ready;
        std::exception_ptr wakeError;
        FiberConstructionCpuPermit* permit = nullptr;
        Task() {
            const long value = sysconf(_SC_PAGESIZE);
            if (value <= 0) throw std::runtime_error("cannot determine case stack page size");
            page = static_cast<size_t>(value);
            bytes = 8 * 1024 * 1024 + 2 * page;
            allocation = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (allocation == MAP_FAILED) throw std::runtime_error("cannot reserve case stack");
            if (mprotect(static_cast<char*>(allocation)+page, bytes-2*page, PROT_READ|PROT_WRITE) != 0) {
                munmap(allocation, bytes);
                allocation = MAP_FAILED;
                throw std::runtime_error("cannot protect case stack");
            }
        }
        ~Task() { if (allocation != MAP_FAILED) munmap(allocation, bytes); }
    };
    struct Worker {
        Impl* owner;
        ucontext_t scheduler{};
        Task* task = nullptr;
        std::vector<std::unique_ptr<Task>> contexts;
    };
    struct Batch {
        size_t next = 0, count = 0;
        std::atomic<size_t> remaining{0};
        std::function<void(size_t)> work;
    };
    static thread_local Worker* current;
    // Threads must join before any shared jobs, contexts or error state die.
    std::vector<std::unique_ptr<Worker>> states;
    std::mutex errorMutex, jobsMutex;
    std::exception_ptr error;
    std::atomic<bool> cancelled{false};
    std::atomic<size_t> nextRoot{0};
    std::atomic<size_t> remainingRoots;
    size_t roots;
    std::function<void()> rootWork;
    std::deque<std::shared_ptr<Batch>> batches;
    std::vector<std::jthread> threads;

    Impl(size_t workers, size_t cases, std::function<void()> work, size_t contexts)
        : remainingRoots(cases), roots(cases), rootWork(std::move(work)) {
        if (!workers || !contexts || !rootWork) throw std::invalid_argument("empty case worker/context/work budget");
        if (!cases) return;
        // Allocate before launching any threads, so allocation failure cannot
        // leave workers waiting on work owned by a half-constructed queue.
        for (size_t i=0;i<workers;++i) {
            auto state=std::make_unique<Worker>();
            state->owner=this;
            for(size_t j=0;j<contexts;++j) state->contexts.push_back(std::make_unique<Task>());
            states.push_back(std::move(state));
        }
        try {
            for(auto& state:states) threads.emplace_back([this,p=state.get()] { run(*p); });
        } catch (...) {
            fail(std::current_exception());
            threads.clear();
            throw;
        }
    }
    ~Impl() {
        if(!threads.empty()) {
            fail(std::make_exception_ptr(std::runtime_error("fiber case queue destroyed")));
            threads.clear();
        }
    }
    void fail(std::exception_ptr exception) {
        { std::lock_guard lock(errorMutex); if(!error) error=exception; }
        cancelled.store(true, std::memory_order_release);
        FiberCaseQueue::notify();
    }
    void check() {
        if(!cancelled.load(std::memory_order_acquire)) return;
        std::lock_guard lock(errorMutex);
        std::rethrow_exception(error);
    }
    bool hasChild() {
        std::lock_guard lock(jobsMutex);
        return !batches.empty();
    }
    bool takeChild(std::function<void()>& work, const std::shared_ptr<Batch>& own = {}) {
        std::shared_ptr<Batch> batch;
        size_t index;
        {
            std::lock_guard lock(jobsMutex);
            auto found=own ? std::find(batches.begin(),batches.end(),own) : batches.begin();
            if(found==batches.end()) return false;
            batch=*found;
            index=batch->next++;
            if(batch->next==batch->count) batches.erase(found);
        }
        work=[this,batch,index] {
            try { check(); batch->work(index); }
            catch(...) { fail(std::current_exception()); }
            --batch->remaining;
            FiberCaseQueue::notify();
        };
        return true;
    }
    static void entry() {
        auto& w=*current;
        auto& task=*w.task;
        while(true) {
            try { task.work(); }
            catch(...) { w.owner->fail(std::current_exception()); }
            task.work={};
            if(!task.child) --w.owner->remainingRoots;
            task.done=true;
            FiberCaseQueue::notify();
            // Persistent contexts are initialized once on their owning thread.
            // An OS context-switch failure cannot safely unwind this stack.
            if(swapcontext(&task.context,&w.scheduler)!=0) std::terminate();
        }
    }
    static void await(std::function<bool()> ready, bool interruptible) {
        auto& w=*current;
        auto& task=*w.task;
        if(interruptible) w.owner->check();
        if(ready()) return;
        task.ready=std::move(ready);
        task.interruptible=interruptible;
        if(swapcontext(&task.context,&w.scheduler)!=0) std::terminate();
        task.ready={};
        task.interruptible=true;
        if(task.wakeError) {
            auto error=std::exchange(task.wakeError,{});
            std::rethrow_exception(error);
        }
        // If ready() acquired a permit, return so its owner records acquisition
        // before any exception; cancellation wins only before predicate success.
        if(task.cancelledWake) w.owner->check();
    }
    void run(Worker& w) {
        try {
            for(auto& task:w.contexts) {
                if(getcontext(&task->context)!=0) throw std::runtime_error("cannot create case context");
                task->context.uc_stack.ss_sp=static_cast<char*>(task->allocation)+task->page;
                task->context.uc_stack.ss_size=task->bytes-2*task->page;
                task->context.uc_link=nullptr;
                makecontext(&task->context,&Impl::entry,0);
            }
        } catch(...) { fail(std::current_exception()); return; }
        while(true) {
            const auto observed=epoch.load();
            bool progressed=false;
            // Resume ready parents first, then children; admit work only into
            // completed contexts. Child jobs precede unrelated root cases.
            for(int phase=0;phase<3;++phase) for(auto& pointer:w.contexts) {
                auto& task=*pointer;
                if(phase<2 && (task.done || task.child!=(phase==1))) continue;
                if(phase==2 && !task.done) continue;
                if(task.done) {
                    task.child=takeChild(task.work);
                    if(!task.child) {
                        if(cancelled.load() || nextRoot.fetch_add(1)>=roots) continue;
                        task.work=rootWork;
                    }
                    task.done=false;
                }
                if(task.ready) {
                    task.cancelledWake=false;
                    if(task.interruptible && cancelled.load()) {
                        // Resume to unwind at the await call.
                        task.cancelledWake=true;
                    } else {
                        try { if(!task.ready()) continue; }
                        catch(...) { task.wakeError=std::current_exception(); }
                    }
                }
                progressed=true;
                w.task=&task;
                current=&w;
                if(swapcontext(&w.scheduler,&task.context)!=0) std::terminate();
                current=nullptr;
            }
            bool pending=false;
            for(const auto& task:w.contexts) pending |= !task->done;
            if(!pending && !hasChild() && (cancelled.load() || remainingRoots.load()==0)) break;
            if(!progressed) {
                std::unique_lock lock(wakeMutex);
                wake.wait_for(lock,std::chrono::milliseconds(10),[&] { return epoch.load()!=observed; });
            }
        }
    }
};
thread_local FiberCaseQueue::Impl::Worker* FiberCaseQueue::Impl::current=nullptr;
#else
class FiberCaseQueue::Impl {};
#endif

bool FiberCaseQueue::supported() noexcept {
#ifdef VC_CASE_CONTEXT_SUPPORTED
    return true;
#else
    return false;
#endif
}
bool FiberCaseQueue::active() noexcept {
#ifdef VC_CASE_CONTEXT_SUPPORTED
    return Impl::current!=nullptr;
#else
    return false;
#endif
}
void FiberCaseQueue::notify() { ++epoch; wake.notify_all(); }
FiberConstructionCpuPermit*& FiberCaseQueue::constructionPermit() {
#ifdef VC_CASE_CONTEXT_SUPPORTED
    if(active()) return Impl::current->task->permit;
#endif
    return outsidePermit;
}
FiberCaseQueue::FiberCaseQueue(size_t workers,size_t cases,std::function<void()> work,size_t contexts) {
#ifdef VC_CASE_CONTEXT_SUPPORTED
    if(active()) throw std::logic_error("use parallelFor for nested case work");
    impl_=std::make_unique<Impl>(workers,cases,std::move(work),contexts);
#else
    (void)workers; (void)cases; (void)work; (void)contexts;
    throw std::runtime_error("fiber case scheduling requires Linux/macOS contexts");
#endif
}
FiberCaseQueue::~FiberCaseQueue()=default;
void FiberCaseQueue::cancel(const std::string& reason) {
#ifdef VC_CASE_CONTEXT_SUPPORTED
    impl_->fail(std::make_exception_ptr(std::runtime_error(reason.empty()?"fiber cases cancelled":reason)));
#endif
}
void FiberCaseQueue::join() {
#ifdef VC_CASE_CONTEXT_SUPPORTED
    if(active()) throw std::logic_error("cannot join a case queue from a case");
    impl_->threads.clear();
    impl_->check();
#endif
}
void FiberCaseQueue::await(std::function<bool()> ready) {
    if(!active()) throw std::logic_error("case await outside scheduler");
#ifdef VC_CASE_CONTEXT_SUPPORTED
    Impl::await(std::move(ready),true);
#endif
}
void FiberCaseQueue::parallelFor(size_t count,const std::function<void(size_t)>& work) {
    if(!count) return;
    if(!work) throw std::invalid_argument("empty child work");
    if(!active()) { for(size_t i=0;i<count;++i) work(i); return; }
#ifdef VC_CASE_CONTEXT_SUPPORTED
    auto& owner=*Impl::current->owner;
    owner.check();
    auto batch=std::make_shared<Impl::Batch>();
    batch->count=count; batch->remaining=count; batch->work=work;
    { std::lock_guard lock(owner.jobsMutex); owner.batches.push_back(batch); }
    notify();
    // Parents help from their existing stack even when every context is full.
    // Drain ALL children before unwinding captured references on cancellation.
    while(batch->remaining.load()) {
        std::function<void()> child;
        if(owner.takeChild(child,batch)) child();
        else Impl::await([&] { return !batch->remaining.load(); },false);
    }
    owner.check();
#endif
}
} // namespace vc::fiber_tracer
