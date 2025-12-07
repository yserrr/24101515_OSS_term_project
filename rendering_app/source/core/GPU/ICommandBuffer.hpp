#ifndef MYPROJECT_ICOMMANDBUFFER_H
#define MYPROJECT_ICOMMANDBUFFER_H

class ICommandBuffer
{
  public:
  virtual void cmdSetViewports() =0;
  virtual void cmdBeginRendering() =0;
  virtual void cmdDraw() =0;
  virtual void cmdDrawQuad() =0;
  virtual void cmdCopyBufferToBuffer() =0;
  virtual void cmdDispatch() =0;
  virtual void cmdCopyBufferToImage() =0;
};


#endif //MYPROJECT_ICOMMANDBUFFER_H
