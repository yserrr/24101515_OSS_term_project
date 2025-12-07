//
// Created by dlwog on 25. 10. 22..
//

#ifndef MYPROJECT_BUFFER_H
#define MYPROJECT_BUFFER_H

enum class BufferType
{
  VERTEX,
  INDEX,
  UNIFORM,
  STORAGE,
  STAGE
};

class IBuffer
{
  virtual ~IBuffer() = 0;
  virtual void loadData() =0;
  virtual void create() =0;
};


#endif //MYPROJECT_BUFFER_H
