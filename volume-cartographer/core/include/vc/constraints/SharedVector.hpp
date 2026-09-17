#pragma once
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace vc::constraints {
// Value semantics for graph snapshots, with storage shared until a writer asks
// for mutable access. Never retain mutable iterators across a snapshot copy.
template<class T> class SharedVector {
    std::shared_ptr<std::vector<T>> values_ = std::make_shared<std::vector<T>>();
    std::vector<T>& write() {
        if(!values_.unique())values_=std::make_shared<std::vector<T>>(*values_);
        return *values_;
    }
public:
    using value_type=T;
    using iterator=typename std::vector<T>::iterator;
    using const_iterator=typename std::vector<T>::const_iterator;
    SharedVector()=default;
    SharedVector(std::vector<T> values):values_(std::make_shared<std::vector<T>>(std::move(values))){}
    SharedVector(std::initializer_list<T> values):SharedVector(std::vector<T>(values)){}
    const std::vector<T>& read()const{return *values_;}
    operator const std::vector<T>&()const{return read();}
    bool sharesStorage(const SharedVector& other)const{return values_==other.values_;}
    size_t size()const{return values_->size();}
    size_t capacity()const{return values_->capacity();}
    bool empty()const{return values_->empty();}
    const T& operator[](size_t i)const{return (*values_)[i];}
    T& operator[](size_t i){return write()[i];}
    const T& at(size_t i)const{return values_->at(i);}
    T& at(size_t i){return write().at(i);}
    const T& front()const{return values_->front();}
    T& front(){return write().front();}
    const T& back()const{return values_->back();}
    T& back(){return write().back();}
    const T* data()const{return values_->data();}
    T* data(){return write().data();}
    const_iterator begin()const{return values_->begin();}
    const_iterator end()const{return values_->end();}
    const_iterator cbegin()const{return values_->cbegin();}
    const_iterator cend()const{return values_->cend();}
    iterator begin(){return write().begin();}
    iterator end(){return write().end();}
    void clear(){values_=std::make_shared<std::vector<T>>();}
    void reserve(size_t n){write().reserve(n);}
    void resize(size_t n){write().resize(n);}
    void push_back(const T& value){write().push_back(value);}
    void push_back(T&& value){write().push_back(std::move(value));}
    template<class It> iterator insert(const_iterator pos,It first,It last){
        const auto offset=pos-values_->cbegin();auto& v=write();return v.insert(v.cbegin()+offset,first,last);
    }
    iterator erase(const_iterator first,const_iterator last){
        const auto a=first-values_->cbegin(),b=last-values_->cbegin();auto& v=write();return v.erase(v.cbegin()+a,v.cbegin()+b);
    }
    iterator erase(const_iterator pos){return erase(pos,pos+1);}
    template<class Predicate> void eraseIf(Predicate predicate){auto& v=write();std::erase_if(v,std::move(predicate));}
    friend bool operator==(const SharedVector& a,const SharedVector& b){return a.values_==b.values_||a.read()==b.read();}
};
}
