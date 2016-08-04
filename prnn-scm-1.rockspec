package = "prnn"
version = "scm-1"

source = {
   url = "https://github.com/iamalbert/prnn.torch",
}

description = {
   summary = "Torch7 FFI binding for Baidu PRNN",
   detailed = [[
   All PRNN modules exposed as nn.Module derivatives so
   that they can be used with torch's neural network package
   ]],
   homepage = "https://github.com/iamalbert/prnn.torch",
   license = "MIT"
}

dependencies = {
   "torch >= 7.0",
   "cutorch",
   "nn"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install",
   copy_directories = {
      "test"
   }

}
