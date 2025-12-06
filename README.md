24101515_OSS_term_project
오소소 텀프로젝트 repo입니다. 

Project Name: Vulkan study
1학기동안 oss term project를 진행하였습니다. 


C++ Compiler: A C++ compiler that supports C++11 or later 
CMake: For building the project.
KTX Library: The Khronos KTX library (likely already included as a submodule).
Python (Optional): Required if the AI project portion utilizes Python scripts.
Pybind11 (Optional): Required for C++/Python interoperability if used.
[Other dependencies should be listed here as they become apparent from the source code]
Installation & Setup Instructions
Clone the repository:

git clone https://github.com/yserrr/24101515_OSS_term_project.git
cd 24101515_OSS_term_project
Initialize submodules (if any):

git submodule init
git submodule update
Build the project using CMake:

mkdir build
cd build
cmake ..
make # or mingw32-make or nmake
Windows: If using Visual Studio, you can open the generated .sln file.
[Additional steps, specific to running the AI project, should be added here once those are identified. For example: Setting up a virtual environment for Python if required. Installing Python packages required for the AI project. Configuring environment variables.]

Usage Examples & API Documentation
[Provide code examples of how to use the sculpting application or the AI project, OR link to a separate documentation file.]

[Example for Sculpting App:]

To run the sculpting application:

./sculpting_app
[Example for AI Project assuming a Python script:]

python AI_Project/main.py
Configuration Options
[Explain any configuration options the user might need, such as environment variables, command line arguments, or configuration files. For example:]

Environment Variables: [Specific environment variables used by the AI project and their purposes]
Command Line Arguments: [Command line arguments for the sculpting app and AI project]
Configuration Files: [Location and format of configuration files, and what they configure.]
Contributing Guidelines
We welcome contributions! Here are the guidelines:

Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes and commit them with clear, descriptive commit messages.
Submit a pull request.
License Information
[Since no license is specified, provide a default suggestion. If a license IS found in the source code, change this section to reflect the ACTUAL license.]

This project is currently unlicensed. You are free to view and experiment with the code, but you do not have permission to distribute, modify, or use it commercially.

If you wish to use the KTX library, please refer to the sculping_app/extern/ktx/LICENSE.md file for its licensing terms (Apache 2.0).

Acknowledgments
Khronos Group for the KTX library.
[Any other third-party libraries or resources used in the project.]
