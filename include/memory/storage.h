#ifndef __MINIDL_STORAGE_H__
#define __MINIDL_STORAGE_H__

#include <memory>

#include "../basicTypes.h"
namespace miniDL {
class MemoryPool;

class Storage {
   private:
    void* _ptr;  // point to data
    size_t _size;
    Device _dev;

    MemoryPool* _pool;

   public:
    Storage(size_t size, Device dev);
    ~Storage();

    // cann't copy
    Storage(const Storage&)            = delete;
    Storage& operator=(const Storage&) = delete;

    // allow move
    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;

    static std::shared_ptr<Storage> create(size_t size, Device dev);

    std::shared_ptr<Storage> toDevice(Device dev);

    void* data() { return _ptr; }
    const void* data() const { return _ptr; }
    Device device() const { return _dev; }
    size_t size() const {return _size;}
};
}  // namespace miniDL
#endif  // __MINIDL_STORAGE_H__