//
// Created by ljh on 25. 10. 16..
//

#ifndef MYPROJECT_UNIQUE_BUILD_HPP
#define MYPROJECT_UNIQUE_BUILD_HPP
#include <memory>

namespace mns{
  template<typename UPTR>
  using uptr = std::unique_ptr<UPTR>;

  template<typename UPTR>
  uptr<UPTR> mUptr()
    {
    return std::make_unique<UPTR>();
    }
  template<typename UPTR>
 uptr<UPTR> mUptr(UPTR&& obj)
  {
    return std::make_unique<UPTR>(std::move(obj));
  }

}
#endif //MYPROJECT_UNIQUE_BUILD_HPP