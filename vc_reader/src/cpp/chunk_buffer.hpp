#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <iterator>
#include <span>
#include <stdexcept>
#include <variant>
#include <vector>

namespace vcr {

// Owned contiguous bytes. Ordinary constructors initialize their contents;
// Overwrite is reserved for decoders that fill every byte before publication.
// Moving a source vector adopts its allocation without copying the payload.
class Chunk {
    using Vector=std::vector<std::uint8_t>;
    std::variant<Vector,std::unique_ptr<std::uint8_t[]>> storage_;
    std::size_t size_=0;
public:
    struct Overwrite {};
    using value_type=std::uint8_t;
    using iterator=value_type*;
    using const_iterator=const value_type*;
    Chunk()=default;
    explicit Chunk(std::size_t size):Chunk(size,std::uint8_t{}){}
    Chunk(std::size_t size,std::uint8_t value):storage_(Vector(size,value)),size_(size){}
    Chunk(Overwrite,std::size_t size)
        :storage_(std::make_unique_for_overwrite<std::uint8_t[]>(size)),size_(size){}
    Chunk(Vector bytes):storage_(std::move(bytes)),size_(std::get<Vector>(storage_).size()){}
    template<std::input_iterator It> Chunk(It first,It last):Chunk(Vector(first,last)){}
    Chunk(const Chunk& other):Chunk(other.begin(),other.end()){}
    Chunk(Chunk&& other)noexcept:storage_(std::move(other.storage_)),size_(other.size_){other.size_=0;}
    Chunk& operator=(const Chunk& other){if(this!=&other){Chunk copy(other);*this=std::move(copy);}return *this;}
    Chunk& operator=(Chunk&& other)noexcept {
        if(this!=&other){storage_=std::move(other.storage_);size_=other.size_;other.size_=0;}return *this;
    }
    std::size_t size()const noexcept{return size_;}
    bool empty()const noexcept{return !size_;}
    value_type* data()noexcept {
        if(auto* bytes=std::get_if<Vector>(&storage_))return bytes->data();
        return std::get<std::unique_ptr<value_type[]>>(storage_).get();
    }
    const value_type* data()const noexcept{return const_cast<Chunk*>(this)->data();}
    iterator begin()noexcept{return data();}
    iterator end()noexcept{return size_?data()+size_:data();}
    const_iterator begin()const noexcept{return data();}
    const_iterator end()const noexcept{return size_?data()+size_:data();}
    value_type& operator[](std::size_t i)noexcept{return data()[i];}
    const value_type& operator[](std::size_t i)const noexcept{return data()[i];}
    value_type& at(std::size_t i){if(i>=size_)throw std::out_of_range("chunk byte index");return data()[i];}
    const value_type& at(std::size_t i)const {if(i>=size_)throw std::out_of_range("chunk byte index");return data()[i];}
    friend bool operator==(const Chunk& a,const Chunk& b){return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin());}
    friend bool operator==(const Chunk& a,const Vector& b){return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin());}
    friend bool operator==(const Vector& a,const Chunk& b){return b==a;}
};
using ChunkPtr=std::shared_ptr<const Chunk>;
} // namespace vcr
