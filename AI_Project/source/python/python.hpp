#ifndef MYPROJECT_PYTHON_HPP
#define MYPROJECT_PYTHON_HPP


#include "extern/pybind11/pybind11.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace py = pybind11;

class Python
{
  void loadMnist();
};
#endif //MYPROJECT_PYTHON_HPP